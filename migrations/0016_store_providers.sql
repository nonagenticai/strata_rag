-- Providers bound per STORE (agent D83): a store may pin its own embed / rerank provider instead
-- of its tenant's. A request acting on a store resolves, per capability: the store's own row →
-- the nearest ancestor's row → the tenant's row (`tenant_providers`, 0006) → the dev default →
-- 409 provider_not_configured. `chat` is not per store: it does not touch stored vectors and stays
-- tenant-wide (0008).
--
-- Same columns and rules as `tenant_providers` (0006) minus the chat-only options. The key is
-- AES-256-GCM under STRATA_PROVIDER_KEY, tenant || store || capability as associated data, so a
-- ciphertext moved to another store, capability or into `tenant_providers` does not decrypt. A
-- row has no tenant column: its store fixes the tenant, and every statement joins `stores` on it.
--
-- `reembed_copy_jobs`: `POST /api/stores/{id}/reembed-copy` re-embeds a store's subtree into an
-- empty target store (in the target's model) in a background task; the job row is what
-- `GET /api/stores/{id}/reembed-copy/{job}` reads, on any replica. A running job beats
-- `heartbeat_at`; one whose beat went stale lost its process and reads as failed `interrupted`.
--
-- Drift-tolerant: IF NOT EXISTS throughout. No `stores` write, so the 0010 trap does not apply.

CREATE TABLE IF NOT EXISTS store_providers (
  store         uuid        NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
  capability    text        NOT NULL,
  base_url      text        NOT NULL,
  dialect       text        NOT NULL,
  model         text        NOT NULL,
  model_id      text,
  dims          int,
  regions       text[]      NOT NULL,
  zdr           bool        NOT NULL,
  timeout_ms    int         NOT NULL,
  concurrency   int         NOT NULL,
  api_key_ct    bytea,
  api_key_nonce bytea,
  key_version   int,
  updated_at    timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (store, capability),
  CONSTRAINT store_providers_capability CHECK (capability IN ('embed', 'rerank')),
  CONSTRAINT store_providers_positive CHECK (timeout_ms > 0 AND concurrency > 0),
  CONSTRAINT store_providers_key_complete CHECK (
    (api_key_ct IS NULL) = (api_key_nonce IS NULL)
    AND (api_key_ct IS NULL) = (key_version IS NULL)
  )
);

CREATE TABLE IF NOT EXISTS reembed_copy_jobs (
  id           uuid        PRIMARY KEY,
  tenant       uuid        NOT NULL,
  source_store uuid        NOT NULL,
  target_store uuid        NOT NULL,
  -- the target's model ("{model}@{dims}") the copy embeds with
  embed_model  text        NOT NULL,
  state        text        NOT NULL,
  chunks_done  int         NOT NULL DEFAULT 0,
  chunks_total int         NOT NULL,
  error        text,
  started_at   timestamptz NOT NULL DEFAULT now(),
  heartbeat_at timestamptz NOT NULL DEFAULT now(),
  finished_at  timestamptz,
  CONSTRAINT reembed_copy_jobs_tenant_not_nil
    CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid),
  CONSTRAINT reembed_copy_jobs_distinct CHECK (source_store <> target_store),
  CONSTRAINT reembed_copy_jobs_progress
    CHECK (chunks_done >= 0 AND chunks_done <= chunks_total),
  CONSTRAINT reembed_copy_jobs_state CHECK (
    (state = 'running' AND finished_at IS NULL AND error IS NULL)
    OR (state = 'done' AND finished_at IS NOT NULL AND error IS NULL)
    OR (state = 'failed' AND finished_at IS NOT NULL AND error IS NOT NULL)
  )
);

-- one copy into a target at a time, across every replica
CREATE UNIQUE INDEX IF NOT EXISTS reembed_copy_jobs_one_running_per_target
  ON reembed_copy_jobs (target_store) WHERE state = 'running';
CREATE INDEX IF NOT EXISTS reembed_copy_jobs_tenant_idx ON reembed_copy_jobs (tenant, id);
