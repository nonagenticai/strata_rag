-- Running content sums of a leaf, so an ingest or a delete stops rescanning the whole leaf.
--
-- Until now `db::stores::recompute_node` rebuilt a leaf's rollup from scratch on every write:
-- `avg(embedding)` over every chunk of the leaf and `sum(token_count)` over every document. One
-- ingest therefore cost O(leaf size), and a bulk load of N documents into one leaf O(N²). The leaf
-- now carries the sum of its chunk vectors (float8, so millions of adds and subtracts stay
-- accurate), its chunk count and its document token total, kept by statement-level triggers on
-- `chunks` and `documents`, so EVERY writer keeps them exact: ingest, deletes and source prunes,
-- the FK cascade of a document or store delete, `strata reembed`, the re-embed copy, raw SQL.
-- `recompute_node` reads them: content_vec = normalise(content_sum / content_chunks), the same
-- value the old `avg` gave (the mean, normalised), and token_count / chunk_count as before.
--
-- Statement-level with transition tables: one `stores` UPDATE per statement and leaf, not one per
-- chunk row. Each trigger updates only the leaf the rows live in; every writer already holds that
-- leaf's row lock (ingest and the deletes lock it first), so no new lock order appears. A leaf
-- that drops to zero chunks gets content_sum NULL (no float residue survives an emptied leaf).
-- Routers keep NULL / 0: they roll up their children's stores rows, as before.
--
-- The 0010 trap: this migration ALTERs `stores` first and writes it (the backfill) last, the
-- order that is fine as-is. Note for later migrations: a write to `chunks` or `documents` now
-- also writes `stores`, so it leaves a pending `stores_bump_generation` event too.

ALTER TABLE stores
  ADD COLUMN IF NOT EXISTS content_sum    float8[],
  ADD COLUMN IF NOT EXISTS content_chunks bigint NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS content_tokens bigint NOT NULL DEFAULT 0;

-- a + k·b, element by element; a NULL `a` counts as zeros.
CREATE OR REPLACE FUNCTION strata_float8_axpy(a float8[], b real[], k float8) RETURNS float8[]
LANGUAGE sql IMMUTABLE AS $$
  SELECT array_agg(coalesce(x, 0) + k * coalesce(y, 0) ORDER BY i)
  FROM unnest(a, b) WITH ORDINALITY AS t(x, y, i)
$$;

CREATE OR REPLACE FUNCTION chunks_content_inserted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  UPDATE stores s
     SET content_sum    = strata_float8_axpy(s.content_sum, d.total::real[], 1),
         content_chunks = s.content_chunks + d.n
    FROM (SELECT store_id, sum(embedding) AS total, count(*) AS n
            FROM new_chunks GROUP BY store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION chunks_content_deleted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  -- a store deleted in this transaction (the cascade of a store delete) matches no row here
  UPDATE stores s
     SET content_sum    = CASE WHEN s.content_chunks - d.n <= 0 THEN NULL
                               ELSE strata_float8_axpy(s.content_sum, d.total::real[], -1) END,
         content_chunks = greatest(s.content_chunks - d.n, 0)
    FROM (SELECT store_id, sum(embedding) AS total, count(*) AS n
            FROM old_chunks GROUP BY store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

-- Only a changed embedding moves the sum (a chunk never changes store: documents never move, and
-- the 0001 trigger pins a chunk's store to its document's).
CREATE OR REPLACE FUNCTION chunks_content_updated() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  UPDATE stores s
     SET content_sum = strata_float8_axpy(
                         strata_float8_axpy(s.content_sum, d.added::real[], 1),
                         d.removed::real[], -1)
    FROM (SELECT n.store_id, sum(n.embedding) AS added, sum(o.embedding) AS removed
            FROM new_chunks n JOIN old_chunks o ON o.id = n.id
           WHERE n.embedding IS DISTINCT FROM o.embedding
           GROUP BY n.store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION documents_content_inserted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  UPDATE stores s
     SET content_tokens = s.content_tokens + d.t
    FROM (SELECT store_id, sum(token_count)::bigint AS t FROM new_documents GROUP BY store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION documents_content_deleted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  UPDATE stores s
     SET content_tokens = s.content_tokens - d.t
    FROM (SELECT store_id, sum(token_count)::bigint AS t FROM old_documents GROUP BY store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

-- A provenance refresh (`update_document_source`) leaves token_count alone and writes no store.
CREATE OR REPLACE FUNCTION documents_content_updated() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  UPDATE stores s
     SET content_tokens = s.content_tokens + d.t
    FROM (SELECT n.store_id, sum(n.token_count - o.token_count)::bigint AS t
            FROM new_documents n JOIN old_documents o ON o.id = n.id
           WHERE n.token_count <> o.token_count
           GROUP BY n.store_id) d
   WHERE s.id = d.store_id;
  RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS chunks_content_inserted ON chunks;
CREATE TRIGGER chunks_content_inserted
  AFTER INSERT ON chunks REFERENCING NEW TABLE AS new_chunks
  FOR EACH STATEMENT EXECUTE FUNCTION chunks_content_inserted();
DROP TRIGGER IF EXISTS chunks_content_deleted ON chunks;
CREATE TRIGGER chunks_content_deleted
  AFTER DELETE ON chunks REFERENCING OLD TABLE AS old_chunks
  FOR EACH STATEMENT EXECUTE FUNCTION chunks_content_deleted();
DROP TRIGGER IF EXISTS chunks_content_updated ON chunks;
CREATE TRIGGER chunks_content_updated
  AFTER UPDATE ON chunks REFERENCING OLD TABLE AS old_chunks NEW TABLE AS new_chunks
  FOR EACH STATEMENT EXECUTE FUNCTION chunks_content_updated();

DROP TRIGGER IF EXISTS documents_content_inserted ON documents;
CREATE TRIGGER documents_content_inserted
  AFTER INSERT ON documents REFERENCING NEW TABLE AS new_documents
  FOR EACH STATEMENT EXECUTE FUNCTION documents_content_inserted();
DROP TRIGGER IF EXISTS documents_content_deleted ON documents;
CREATE TRIGGER documents_content_deleted
  AFTER DELETE ON documents REFERENCING OLD TABLE AS old_documents
  FOR EACH STATEMENT EXECUTE FUNCTION documents_content_deleted();
DROP TRIGGER IF EXISTS documents_content_updated ON documents;
CREATE TRIGGER documents_content_updated
  AFTER UPDATE ON documents REFERENCING OLD TABLE AS old_documents NEW TABLE AS new_documents
  FOR EACH STATEMENT EXECUTE FUNCTION documents_content_updated();

-- Backfill from the rows as they are (the triggers above already hold `chunks` and `documents`
-- against concurrent writers until this migration commits). Last: no ALTER of `stores` follows.
UPDATE stores s
   SET content_sum = a.total::real[]::float8[], content_chunks = a.n
  FROM (SELECT store_id, sum(embedding) AS total, count(*) AS n FROM chunks GROUP BY store_id) a
 WHERE s.id = a.store_id;
UPDATE stores s
   SET content_tokens = a.t
  FROM (SELECT store_id, sum(token_count)::bigint AS t FROM documents GROUP BY store_id) a
 WHERE s.id = a.store_id;
