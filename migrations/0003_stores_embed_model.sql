-- Which embedding model made a store's vectors, as "{model}@{dims}" (STRATA_EMBED_MODEL @
-- STRATA_EMBED_DIMS). Every store that exists before this migration was embedded in-process with
-- bge-m3 at 1024 dims. The default only backfills those rows: it is dropped right after, so every
-- new store must state its model. Query and ingest refuse a store whose model differs from the
-- configured one (409 embed_model_mismatch) instead of mixing vector spaces.
ALTER TABLE stores ADD COLUMN embed_model text NOT NULL DEFAULT 'BAAI/bge-m3@1024';
ALTER TABLE stores ALTER COLUMN embed_model DROP DEFAULT;
