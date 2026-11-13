-- Caller-side provenance for a document: the care-bus source it came from (`source_id`), the sync
-- run that last delivered it (`source_run`), and the source's mapped fields (`metadata`, a JSON
-- object). A re-sync ingests every document with a new run and then prunes the source's documents
-- that do not carry it (DELETE /api/sources/{source_id}/documents?except_run=...).
ALTER TABLE documents
  ADD COLUMN source_id  text,
  ADD COLUMN source_run text,
  ADD COLUMN metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD CONSTRAINT documents_metadata_is_object CHECK (jsonb_typeof(metadata) = 'object'),
  -- a run only means something within a source
  ADD CONSTRAINT documents_source_run_needs_source CHECK (source_run IS NULL OR source_id IS NOT NULL);

-- Prune: WHERE source_id = $1 [AND store_id = $3] AND source_run IS DISTINCT FROM $2.
CREATE INDEX documents_source_run_idx ON documents (source_id, source_run)
  WHERE source_id IS NOT NULL;
