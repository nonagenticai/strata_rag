//! `strata reembed` (spec §8.3): re-embed a tenant's whole tree with its CURRENT embed provider.
//!
//! A store records the model that made its vectors (`stores.embed_model`), and query / ingest /
//! re-describe refuse a tree in another model than the configured one (409
//! `embed_model_mismatch`). After a provider switch this is the way out: every chunk and every
//! store's `desc_vec` is embedded again, the content / profile rollups are recomputed from the new
//! vectors, and the tree is tagged with the new model, all in ONE transaction, so a query never
//! sees a tree half in one vector space and half in another.
//!
//! As in ingest, the vectors are computed BEFORE the transaction (no row lock is held during
//! inference). Nothing can add to the tree meanwhile: every write into it is refused with
//! `embed_model_mismatch` while the stores still carry the old model. A store or document deleted
//! meanwhile is caught under the locks and the run fails with a retryable conflict.
//!
//! Memory is bounded by one batch, whatever the tree holds: chunks are read and embedded `batch`
//! at a time and each batch of vectors is staged in `reembed_staging` (migration 0021) under this
//! run's id; the transaction then copies them into `chunks` `batch` rows per UPDATE. Only the
//! chunk ids (16 bytes each) and the store descriptions' vectors are held for the whole run.

use std::collections::{HashMap, HashSet};

use pgvector::Vector;
use uuid::Uuid;

use crate::db::stores::{self, StoreRow};
use crate::engine::{Engine, desc_text};
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;

/// Default `--batch`: chunk texts handed to the embed provider lane per call (the lane itself
/// splits them into `STRATA_BULK_EMBED_BATCH`-sized requests) and rows per UPDATE.
pub const DEFAULT_REEMBED_BATCH: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ReembedReport {
    pub root: Uuid,
    pub stores: usize,
    pub chunks: usize,
    pub embed_model: String,
}

/// `root` and every descendant among `rows`, each with its depth (root = 0).
pub(crate) fn subtree_depths(rows: &[StoreRow], root: Uuid) -> Vec<(Uuid, i32)> {
    let mut children: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for r in rows {
        if let Some(p) = r.parent_id {
            children.entry(p).or_default().push(r.id);
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![(root, 0)];
    while let Some((id, depth)) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        out.push((id, depth));
        for c in children.get(&id).into_iter().flatten() {
            stack.push((*c, depth + 1));
        }
    }
    out
}

/// The tree changed between the embed pass and the transaction (a store or document deleted).
/// Nothing was written; a fresh run starts over from the current rows, so it is retryable.
fn changed_meanwhile(what: String) -> StrataError {
    StrataError::Conflict {
        message: format!("{what} changed during the re-embed; nothing was written, run it again"),
        retryable: true,
    }
}

impl Engine {
    /// Re-embeds `tenant`'s tree under `root` (its root store: a subtree re-embedded below an
    /// ancestor in the old model would mix two vector spaces in that ancestor's rollup) with the
    /// tenant's current embed provider. `batch`: chunk texts per embed call and rows per UPDATE.
    pub async fn reembed_tree(
        &self,
        tenant: Tenant,
        root: Uuid,
        batch: usize,
    ) -> Result<ReembedReport> {
        if batch == 0 {
            return Err(StrataError::InvalidInput(
                "batch must be at least 1".to_string(),
            ));
        }
        let rows = stores::list_stores(&self.pool, tenant).await?;
        let root_row = rows
            .iter()
            .find(|r| r.id == root)
            .ok_or_else(|| StrataError::NotFound(format!("store {root}")))?;
        if root_row.parent_id.is_some() {
            return Err(StrataError::InvalidInput(format!(
                "store {root} is not a root: reembed re-embeds a whole tree, from its root"
            )));
        }
        // a published platform pack is frozen (checked again under the tree lock)
        crate::db::platform::assert_writable(&self.pool, root).await?;
        // the root's providers: its own store binding, else the tenant's
        let inf = self.inference_for(tenant, Some(root)).await?;
        let model = inf.embed_model_id()?.to_string();

        let members = subtree_depths(&rows, root);
        let by_id: HashMap<Uuid, &StoreRow> = rows.iter().map(|r| (r.id, r)).collect();
        let run = Uuid::now_v7();
        // rows of a run whose process died before it could clean up
        sqlx::query("DELETE FROM reembed_staging WHERE created_at < now() - interval '7 days'")
            .execute(&self.pool)
            .await?;
        let done = self
            .reembed_staged(tenant, root, batch, run, &members, &by_id, &model, &inf)
            .await;
        if done.is_err() {
            // the transaction rolled back; drop this run's staged vectors (best effort)
            if let Err(e) = sqlx::query("DELETE FROM reembed_staging WHERE run = $1")
                .bind(run)
                .execute(&self.pool)
                .await
            {
                tracing::warn!(%run, error = %e, "reembed: staged vectors not removed");
            }
        }
        let chunks = done?;
        self.cache.invalidate(tenant).await;
        Ok(ReembedReport {
            root,
            stores: members.len(),
            chunks,
            embed_model: model,
        })
    }

    /// The two phases of `reembed_tree` for run `run`: embed and stage every chunk batch by
    /// batch, then one transaction that locks the tree and switches it to the staged vectors.
    /// Returns the number of chunks re-embedded.
    #[allow(clippy::too_many_arguments)]
    async fn reembed_staged(
        &self,
        tenant: Tenant,
        root: Uuid,
        batch: usize,
        run: Uuid,
        members: &[(Uuid, i32)],
        by_id: &HashMap<Uuid, &StoreRow>,
        model: &str,
        inf: &crate::inference::InferenceHandle,
    ) -> Result<usize> {
        let member_ids: Vec<Uuid> = members.iter().map(|(id, _)| *id).collect();

        // 1. vectors, before the transaction. desc_vec as create_store embeds it (query lane).
        let descs: Vec<String> = member_ids
            .iter()
            .map(|id| desc_text(&by_id[id].name, &by_id[id].description))
            .collect();
        let desc_vecs = inf.embed_queries(descs).await?;
        if desc_vecs.len() != member_ids.len() {
            return Err(StrataError::Model(format!(
                "{} embeddings for {} store descriptions",
                desc_vecs.len(),
                member_ids.len()
            )));
        }
        // chunk texts and vectors one batch at a time; the vectors go to the staging table
        let chunk_ids: Vec<Uuid> =
            sqlx::query_scalar("SELECT id FROM chunks WHERE store_id = ANY($1) ORDER BY id")
                .bind(member_ids.as_slice())
                .fetch_all(&self.pool)
                .await?;
        let mut staged = 0usize;
        for ids in chunk_ids.chunks(batch) {
            // a chunk deleted since the id scan is simply not staged (the final count tells)
            let part: Vec<(Uuid, String)> =
                sqlx::query_as("SELECT id, text FROM chunks WHERE id = ANY($1) ORDER BY id")
                    .bind(ids)
                    .fetch_all(&self.pool)
                    .await?;
            if part.is_empty() {
                continue;
            }
            let vecs = inf
                .embed(part.iter().map(|(_, text)| text.clone()).collect())
                .await?;
            if vecs.len() != part.len() {
                return Err(StrataError::Model(format!(
                    "{} embeddings for {} chunks",
                    vecs.len(),
                    part.len()
                )));
            }
            let part_ids: Vec<Uuid> = part.iter().map(|(id, _)| *id).collect();
            let vecs: Vec<Vector> = vecs.into_iter().map(Vector::from).collect();
            sqlx::query(
                "INSERT INTO reembed_staging (run, chunk_id, embedding) \
                 SELECT $1, u.id, u.v FROM unnest($2::uuid[], $3::vector[]) AS u(id, v)",
            )
            .bind(run)
            .bind(part_ids.as_slice())
            .bind(vecs)
            .execute(&self.pool)
            .await?;
            staged += part.len();
            tracing::info!(
                %tenant,
                done = staged,
                total = chunk_ids.len(),
                "reembed: chunks embedded"
            );
        }

        // 2. one transaction: lock the tree child before parent (every writer's order), then
        //    write chunks, desc_vecs and the model, then recompute the rollups bottom-up.
        let mut tx = self.pool.begin().await?;
        let depths: Vec<i32> = members.iter().map(|(_, d)| *d).collect();
        let locked: Vec<(Uuid, String, String)> = sqlx::query_as(
            "SELECT s.id, s.name, s.description FROM stores s \
             JOIN unnest($1::uuid[], $2::int4[]) AS t(id, depth) ON t.id = s.id \
             WHERE s.tenant = $3 \
             ORDER BY t.depth DESC, s.id \
             FOR UPDATE OF s",
        )
        .bind(member_ids.as_slice())
        .bind(depths.as_slice())
        .bind(tenant.id())
        .fetch_all(&mut *tx)
        .await?;
        if locked.len() != member_ids.len() {
            return Err(changed_meanwhile(format!("the tree under {root}")));
        }
        crate::db::platform::assert_writable(&mut *tx, root).await?;
        for (id, name, description) in &locked {
            let read = by_id[id];
            if *name != read.name || *description != read.description {
                return Err(changed_meanwhile(format!("store {id}")));
            }
        }
        // the staged vectors, `batch` rows per UPDATE (the id list is sorted: each batch is a
        // contiguous stretch of the staging table's primary key)
        let mut updated = 0u64;
        for ids in chunk_ids.chunks(batch) {
            updated += sqlx::query(
                "UPDATE chunks c SET embedding = r.embedding \
                 FROM reembed_staging r \
                 WHERE r.run = $1 AND r.chunk_id = ANY($2) AND c.id = r.chunk_id \
                   AND c.store_id = ANY($3)",
            )
            .bind(run)
            .bind(ids)
            .bind(member_ids.as_slice())
            .execute(&mut *tx)
            .await?
            .rows_affected();
        }
        // no chunk of the tree may keep an old vector
        let now: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks WHERE store_id = ANY($1)")
            .bind(member_ids.as_slice())
            .fetch_one(&mut *tx)
            .await?;
        if u64::try_from(now).unwrap_or(u64::MAX) != updated {
            return Err(changed_meanwhile(format!("the chunks under {root}")));
        }
        for (id, desc_vec) in member_ids.iter().zip(desc_vecs) {
            sqlx::query(
                "UPDATE stores SET desc_vec = $2, embed_model = $3, updated_at = now() \
                 WHERE id = $1 AND tenant = $4",
            )
            .bind(id)
            .bind(Vector::from(desc_vec))
            .bind(model)
            .bind(tenant.id())
            .execute(&mut *tx)
            .await?;
        }
        // every vector changed: rebuild the leaves' running sums exactly, then the rollups
        // deepest first (a router aggregates its children's fresh content_vecs)
        stores::rebuild_content_sums(&mut tx, &member_ids).await?;
        for (id, _, _) in &locked {
            stores::recompute_node(&mut tx, tenant, *id, self.cfg.profile_alpha).await?;
        }
        sqlx::query("DELETE FROM reembed_staging WHERE run = $1")
            .bind(run)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(usize::try_from(updated).unwrap_or(usize::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::stores::StoreKind;

    fn row(id: u128, parent: Option<u128>) -> StoreRow {
        StoreRow {
            id: Uuid::from_u128(id),
            tenant: Uuid::from_u128(99),
            parent_id: parent.map(Uuid::from_u128),
            kind: StoreKind::Router,
            name: format!("s{id}"),
            description: String::new(),
            desc_vec: Vec::new(),
            content_vec: None,
            profile_vec: Vec::new(),
            token_count: 0,
            chunk_count: 0,
            embed_model: "m".to_string(),
        }
    }

    #[test]
    fn subtree_depths_walks_the_tree_only() {
        let rows = vec![
            row(1, None),
            row(2, Some(1)),
            row(3, Some(2)),
            row(4, Some(1)),
            row(5, None), // another root: not in 1's tree
            row(6, Some(5)),
        ];
        let mut got = subtree_depths(&rows, Uuid::from_u128(1));
        got.sort();
        let want: Vec<(Uuid, i32)> = [(1, 0), (2, 1), (3, 2), (4, 1)]
            .into_iter()
            .map(|(id, d)| (Uuid::from_u128(id), d))
            .collect();
        assert_eq!(got, want);
    }
}
