//! Providers bound per store (agent D83, migration 0016): `GET|PUT|DELETE /api/stores/{id}/providers`.
//!
//! A store's embed / rerank / chat registration (chat since D92, migration 0017) wins over its
//! tenant's for every request acting on the store or below it (`providers::registry`). Its rules are the tenant registration's (validation,
//! live probe, SSRF guard, keys sealed and never returned) plus one about vectors: a store's
//! `embed_model` names the model that made its vectors, and its whole lineage shares it (rollups
//! average child vectors into every ancestor). So an embed registration whose model differs from
//! the store's is accepted only for an EMPTY store (no documents, no children) whose ancestors
//! are in that model already (a root has none): the store is then re-tagged with the new model
//! and its `desc_vec` embedded again by the new provider. Anything else is 409
//! `embed_model_mismatch`; the way to move a store with content to another model is to re-embed
//! it into an empty store of that model (`POST /api/stores/{id}/reembed-copy`).

use pgvector::Vector;
use uuid::Uuid;

use crate::db::providers as rows;
use crate::db::stores::{self, StoreRow};
use crate::engine::{Engine, desc_text};
use crate::error::{Result, StrataError};
use crate::providers::{
    PreparedStoreProviders, PutStoreProviders, STORE_CAPABILITIES, StoreProvidersView,
};
use crate::tenant::Tenant;

/// `store` holds nothing: no child store and no document (`chunk_count` alone would miss a
/// document whose chunks were never written).
async fn is_empty(conn: &mut sqlx::PgConnection, store: &StoreRow) -> Result<bool> {
    if store.chunk_count != 0 || stores::child_count(&mut *conn, store.id).await? != 0 {
        return Ok(false);
    }
    let docs: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM documents WHERE store_id = $1)")
            .bind(store.id)
            .fetch_one(&mut *conn)
            .await?;
    Ok(!docs)
}

impl Engine {
    /// `GET /api/stores/{id}/providers`. Another tenant's store is 404.
    pub async fn store_providers(&self, tenant: Tenant, id: Uuid) -> Result<StoreProvidersView> {
        let store = {
            let mut conn = self.pool.acquire().await?;
            stores::get_store(&mut conn, tenant, id).await?
        };
        self.providers
            .store_view(tenant, id, store.embed_model)
            .await
    }

    /// `PUT /api/stores/{id}/providers`: replaces the store's own provider set (a capability left
    /// out is removed, so the store inherits it again); `DELETE` is the empty set. Validated and
    /// probed like `PUT /api/tenant/providers` (422 with the specific code), then the model rule
    /// (module docs). Answers the stored view.
    pub async fn put_store_providers(
        &self,
        tenant: Tenant,
        id: Uuid,
        body: PutStoreProviders,
    ) -> Result<StoreProvidersView> {
        let current = {
            let mut conn = self.pool.acquire().await?;
            let current = stores::get_store(&mut conn, tenant, id).await?;
            // a published platform pack is frozen; checked before the live probe, again under lock
            crate::db::platform::assert_writable(&mut *conn, id).await?;
            current
        };
        let prepared = self.providers.prepare_store(tenant, id, body).await?;
        let retag = prepared
            .embed_model_id()
            .filter(|model| *model != current.embed_model);
        // the store's own desc_vec in the new model, embedded before any lock is taken
        let desc_vec = match (&retag, prepared.embed_handle()) {
            (Some(model), Some(inf)) => {
                let mut conn = self.pool.acquire().await?;
                self.check_retag(&mut conn, &current, model).await?;
                drop(conn);
                let inf = self.metered(tenant, inf);
                Some(
                    inf.embed_queries(vec![desc_text(&current.name, &current.description)])
                        .await?
                        .remove(0),
                )
            }
            _ => None,
        };
        let written = crate::db::retry_once(|| {
            self.put_store_providers_tx(
                tenant,
                &current,
                &prepared,
                retag.as_deref(),
                desc_vec.as_deref(),
            )
        })
        .await;
        // also on failure: a partly applied write must not leave a stale client behind
        self.providers.forget_store(id);
        written?;
        if retag.is_some() {
            // the routing tree caches each store's embed_model and profile_vec
            self.cache.invalidate(tenant).await;
        }
        tracing::info!(
            %tenant,
            store = %id,
            embed = prepared.set.iter().any(|(c, _, _)| c.as_str() == "embed"),
            rerank = prepared.set.iter().any(|(c, _, _)| c.as_str() == "rerank"),
            chat = prepared.set.iter().any(|(c, _, _)| c.as_str() == "chat"),
            retagged = retag.is_some(),
            "store providers replaced"
        );
        self.store_providers(tenant, id).await
    }

    /// 409 `embed_model_mismatch` unless `store` may take `model`: it is empty and every ancestor
    /// already holds `model`.
    async fn check_retag(
        &self,
        conn: &mut sqlx::PgConnection,
        store: &StoreRow,
        model: &str,
    ) -> Result<()> {
        if !is_empty(&mut *conn, store).await? {
            return Err(StrataError::EmbedModelMismatch {
                store_id: store.id,
                store_model: store.embed_model.clone(),
                configured: model.to_string(),
            });
        }
        if let Some(parent) = store.parent_id {
            self.check_lineage_embed_model(&mut *conn, parent, model)
                .await?;
        }
        Ok(())
    }

    async fn put_store_providers_tx(
        &self,
        tenant: Tenant,
        read: &StoreRow,
        prepared: &PreparedStoreProviders,
        retag: Option<&str>,
        desc_vec: Option<&[f32]>,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let locked = stores::get_store_for_update(&mut tx, tenant, read.id).await?;
        crate::db::platform::assert_writable(&mut *tx, read.id).await?;
        if locked.embed_model != read.embed_model
            || locked.name != read.name
            || locked.description != read.description
        {
            return Err(StrataError::Conflict {
                message: format!(
                    "store {} was changed concurrently; retry the request",
                    read.id
                ),
                retryable: false,
            });
        }
        for cap in STORE_CAPABILITIES {
            match prepared.set.iter().find(|(c, _, _)| *c == cap) {
                Some((_, spec, key)) => {
                    rows::store_upsert(&mut tx, read.id, cap, spec, key.as_ref()).await?
                }
                None => rows::store_delete(&mut tx, read.id, cap).await?,
            }
        }
        if let (Some(model), Some(desc_vec)) = (retag, desc_vec) {
            // under the lock: nothing was added since the check, the ancestors still match
            self.check_retag(&mut tx, &locked, model).await?;
            sqlx::query(
                "UPDATE stores SET desc_vec = $2, embed_model = $3, updated_at = now() \
                 WHERE id = $1 AND tenant = $4",
            )
            .bind(read.id)
            .bind(Vector::from(desc_vec.to_vec()))
            .bind(model)
            .bind(tenant.id())
            .execute(&mut *tx)
            .await?;
            // its profile_vec (and the ancestors' rollups) from the new desc_vec
            stores::recompute_upwards(&mut tx, tenant, read.id, self.cfg.profile_alpha).await?;
        }
        tx.commit().await?;
        Ok(())
    }
}
