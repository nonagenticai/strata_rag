-- Tenant eval questions and stored eval runs (POST /api/eval with `source: "tenant"`).
--
-- Tenancy follows 0005/0006: every row carries its tenant (never the nil uuid) and every
-- statement in src/db/eval.rs filters on it, so another tenant's question or run id answers 404
-- exactly like an id that does not exist.
--
-- Drift-tolerant: IF NOT EXISTS throughout, so a database that already holds these objects (a
-- hand-applied copy, a re-run after a partial failure) migrates instead of failing.

-- A tenant's golden question: the query and what a good result is. A result counts as a hit
-- when it matches ANY of the expectations given (at least one is required):
--   expected_document      the document's title, its `source`, or the file name of its source
--   expected_document_ids  one of these document ids
--   expected_chunk_ids     one of these chunk ids
--   expected_text          the chunk text contains it (case- and whitespace-insensitive)
-- expected_leaf (a leaf store id, name path `a/b/c`, or unique leaf name) only feeds leaf accuracy.
CREATE TABLE IF NOT EXISTS eval_questions (
  id                    uuid        PRIMARY KEY DEFAULT uuidv7(),
  tenant                uuid        NOT NULL,
  query                 text        NOT NULL,
  expected_document     text,
  expected_leaf         text,
  expected_document_ids uuid[]      NOT NULL DEFAULT '{}',
  expected_chunk_ids    uuid[]      NOT NULL DEFAULT '{}',
  expected_text         text,
  -- the caller's X-Actor-Id when it sent one
  created_by            uuid,
  -- now(): one value per transaction, so a new row's created_at = updated_at, and an import's
  -- rows share it (their uuidv7 ids, monotonic within a session, keep the file order)
  created_at            timestamptz NOT NULL DEFAULT now(),
  updated_at            timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT eval_questions_tenant_not_nil
    CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid),
  CONSTRAINT eval_questions_query_not_blank CHECK (btrim(query) <> ''),
  CONSTRAINT eval_questions_has_expectation CHECK (
    expected_document IS NOT NULL
    OR expected_text IS NOT NULL
    OR cardinality(expected_document_ids) > 0
    OR cardinality(expected_chunk_ids) > 0
  )
);

-- One question text per tenant (case-insensitive); also the tenant's list order.
CREATE UNIQUE INDEX IF NOT EXISTS eval_questions_tenant_query_key
  ON eval_questions (tenant, lower(query));
CREATE INDEX IF NOT EXISTS eval_questions_tenant_created_idx
  ON eval_questions (tenant, created_at, id);

-- One completed eval run. `questions` is a snapshot of what was evaluated (a question edited or
-- deleted later does not change an old run); `reports` the per-mode recall@1/5/10 rows.
CREATE TABLE IF NOT EXISTS eval_runs (
  id          uuid        PRIMARY KEY DEFAULT uuidv7(),
  tenant      uuid        NOT NULL,
  -- tenant: the tenant's stored questions; inline: questions sent in the request body
  source      text        NOT NULL,
  root_id     uuid,
  modes       text[]      NOT NULL,
  beam        integer     NOT NULL,
  min_score   real        NOT NULL,
  questions   jsonb       NOT NULL,
  reports     jsonb       NOT NULL,
  verdict     text        NOT NULL,
  started_by  uuid,
  started_at  timestamptz NOT NULL,
  finished_at timestamptz NOT NULL,
  CONSTRAINT eval_runs_tenant_not_nil
    CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid),
  CONSTRAINT eval_runs_source CHECK (source IN ('tenant', 'inline')),
  CONSTRAINT eval_runs_verdict CHECK (verdict IN ('PASS', 'FAIL'))
);

CREATE INDEX IF NOT EXISTS eval_runs_tenant_started_idx
  ON eval_runs (tenant, started_at DESC, id DESC);

-- Per question, per mode: the 1-based rank of the first hit (NULL = not in the top 10) and
-- whether routing reached the expected leaf (NULL = flat mode, or no expected_leaf).
-- question_id is the stored question's id at run time (NULL for an inline question); no foreign
-- key, so deleting a question keeps its history.
CREATE TABLE IF NOT EXISTS eval_run_results (
  run_id      uuid    NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
  tenant      uuid    NOT NULL,
  idx         integer NOT NULL,
  mode        text    NOT NULL,
  question_id uuid,
  rank        integer,
  leaf_hit    boolean,
  PRIMARY KEY (run_id, idx, mode),
  CONSTRAINT eval_run_results_rank CHECK (rank IS NULL OR rank >= 1)
);
