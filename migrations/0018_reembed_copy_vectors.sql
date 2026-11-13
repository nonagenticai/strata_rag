-- `POST /api/stores/{id}/reembed-copy {copy_vectors: true}` (agent B3): when the target resolves
-- to the source's embed model, the copy reuses the source's vectors (chunk embeddings and store
-- description vectors) instead of embedding again, so a rerank- or chat-only change gets its own
-- store tree without GPU cost. The job records which kind of copy it is.
--
-- Drift-tolerant: IF NOT EXISTS. No `stores` write, so the 0010 trap does not apply.
ALTER TABLE reembed_copy_jobs ADD COLUMN IF NOT EXISTS copy_vectors bool NOT NULL DEFAULT false;
