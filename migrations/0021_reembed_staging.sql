-- `strata reembed` (src/reembed.rs) without holding the whole tree in memory. Each batch of new
-- chunk vectors is written here as soon as it is embedded (outside any transaction, no row lock
-- held during inference); the final transaction then copies them into `chunks` batch by batch and
-- deletes the run's rows, so the tree still switches to the new vectors all at once. A run's rows
-- are removed when it commits or fails; rows of a process that died mid-run are purged by a later
-- run once they are a week old.
--
-- UNLOGGED: staged vectors are scratch. A Postgres crash empties the table, and the run that
-- staged them fails on its final count check (nothing written) and is started again.
--
-- No `stores` write, so the 0010 trap does not apply.
CREATE UNLOGGED TABLE IF NOT EXISTS reembed_staging (
  run        uuid        NOT NULL,
  chunk_id   uuid        NOT NULL,
  embedding  vector      NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run, chunk_id)
);
CREATE INDEX IF NOT EXISTS reembed_staging_created_idx ON reembed_staging (created_at);
