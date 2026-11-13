-- eval_run_results.tenant must be its run's tenant, and never the nil uuid (0012 left both
-- unchecked: a result row could carry any tenant, unrelated to the run it belongs to).
--
-- A new migration rather than an edit of 0012, per migrations/README.md (sqlx checksums every
-- applied file). Drift-tolerant: each constraint is added only when a constraint of that name is
-- not there yet, and the old single-column FK is dropped IF EXISTS, so a database that already
-- holds any of these migrates instead of failing.

-- The nil tenant is never a tenant (as in eval_questions / eval_runs).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'eval_run_results'::regclass AND conname = 'eval_run_results_tenant_not_nil'
  ) THEN
    ALTER TABLE eval_run_results ADD CONSTRAINT eval_run_results_tenant_not_nil
      CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid);
  END IF;
END $$;

-- The target of the composite FK below (id alone is already unique; this makes (id, tenant) a key).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'eval_runs'::regclass AND conname = 'eval_runs_id_tenant_key'
  ) THEN
    ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_id_tenant_key UNIQUE (id, tenant);
  END IF;
END $$;

-- A result row's (run_id, tenant) must be a run's (id, tenant): its tenant is its run's tenant.
-- It replaces the run_id-only FK (same ON DELETE CASCADE), which it makes redundant.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'eval_run_results'::regclass AND conname = 'eval_run_results_run_tenant_fkey'
  ) THEN
    ALTER TABLE eval_run_results ADD CONSTRAINT eval_run_results_run_tenant_fkey
      FOREIGN KEY (run_id, tenant) REFERENCES eval_runs (id, tenant) ON DELETE CASCADE;
  END IF;
END $$;

ALTER TABLE eval_run_results DROP CONSTRAINT IF EXISTS eval_run_results_run_id_fkey;
