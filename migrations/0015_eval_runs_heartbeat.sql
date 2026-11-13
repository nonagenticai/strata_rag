-- Liveness of a running tenant eval: its worker stamps `heartbeat_at = now()` together with its
-- progress (about once a second). A run is swept `failed` / `interrupted` only once its heartbeat
-- is older than a stale threshold (src/db/eval.rs `EVAL_STALE_AFTER`), not on every service start:
-- during a rolling update the old pod is still finishing its run while the new pod boots.
--
-- A new migration rather than an edit of 0014, per migrations/README.md. Drift-tolerant:
-- ADD COLUMN IF NOT EXISTS, and the backfill only touches rows without a heartbeat. Only an ALTER
-- and an UPDATE of eval_runs (no `stores` write), so the 0010 trap does not apply.

ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS heartbeat_at timestamptz;

-- A run left running before this migration has never beaten: its start is its last sign of life.
UPDATE eval_runs SET heartbeat_at = started_at
 WHERE status = 'running' AND heartbeat_at IS NULL;
