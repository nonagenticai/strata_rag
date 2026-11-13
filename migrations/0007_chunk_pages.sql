-- Page numbers on chunks (spec §8.5): the first and last page a chunk spans, from Bundle's
-- `<!-- page:N -->` anchors. NULL for text without anchors (audio, records, plain text) and for
-- chunks ingested before this migration — re-ingesting a document fills them.
ALTER TABLE chunks
  ADD COLUMN page_start int,
  ADD COLUMN page_end   int;
