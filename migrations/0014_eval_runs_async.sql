-- Asynchronous tenant eval runs: `POST /api/eval` (tenant questions) answers 202 with a run that is
-- `running`, scores it in a background task, and finishes it `completed` (verdict, reports,
-- results) or `failed` (error). A 200-question hybrid run takes minutes, longer than a proxy in
-- front of the API keeps a request open.
--
-- A new migration rather than an edit of 0012/0013, per migrations/README.md. Drift-tolerant:
-- ADD COLUMN IF NOT EXISTS, DROP NOT NULL (a no-op when already nullable), constraints added only
-- when a constraint of that name is not there yet, the index IF NOT EXISTS, and the backfill only
-- touches rows it has not touched yet, so a database that already holds any of this migrates.
--
-- Only ALTERs and an UPDATE of eval_runs (no `stores` write), so the 0010 trap does not apply.

ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'completed';
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS error text;
-- question x mode outcomes scored so far, out of progress_total
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS progress_done integer NOT NULL DEFAULT 0;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS progress_total integer NOT NULL DEFAULT 0;

-- A running or failed run has no verdict / reports; a running one no finished_at.
ALTER TABLE eval_runs ALTER COLUMN verdict DROP NOT NULL;
ALTER TABLE eval_runs ALTER COLUMN reports DROP NOT NULL;
ALTER TABLE eval_runs ALTER COLUMN finished_at DROP NOT NULL;

-- Runs stored before this migration all completed: their progress is complete.
UPDATE eval_runs
   SET progress_total = jsonb_array_length(questions) * cardinality(modes),
       progress_done  = jsonb_array_length(questions) * cardinality(modes)
 WHERE status = 'completed' AND progress_total = 0;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'eval_runs'::regclass AND conname = 'eval_runs_status'
  ) THEN
    ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_status CHECK (
      (status = 'running' AND finished_at IS NULL AND verdict IS NULL AND reports IS NULL
         AND error IS NULL)
      OR (status = 'completed' AND finished_at IS NOT NULL AND verdict IS NOT NULL
         AND reports IS NOT NULL AND error IS NULL)
      OR (status = 'failed' AND finished_at IS NOT NULL AND error IS NOT NULL)
    );
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'eval_runs'::regclass AND conname = 'eval_runs_progress'
  ) THEN
    ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_progress
      CHECK (progress_done >= 0 AND progress_done <= progress_total);
  END IF;
END $$;

-- Single flight per tenant, across every API replica: at most one running run.
CREATE UNIQUE INDEX IF NOT EXISTS eval_runs_one_running
  ON eval_runs (tenant) WHERE status = 'running';
