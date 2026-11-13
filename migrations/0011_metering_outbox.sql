-- Transactional outbox for cost events (spec §8.8; the commonpractice model, its migration 0090).
--
-- Every metered provider call writes one row here; `crate::metering::Drainer` (in `strata serve`)
-- XADDs pending rows to billing's stream (`METERING_STREAM`) and marks them sent. A row is written
-- whether or not Redis is reachable, so an outage, a crash or an OOM after the row is committed
-- loses nothing: the row is delivered when Redis is back.
--
-- Every paid provider call gets a row as the call returns, independent of any business
-- transaction (an ingest or reembed that rolls back or is killed half-way has still recorded the
-- calls it paid for), through an in-memory batch writer (multi-row INSERT). See src/metering.rs.
CREATE TABLE metering_outbox (
  id              uuid        PRIMARY KEY DEFAULT uuidv7(),
  tenant_id       uuid        NOT NULL,
  -- billing's `event_key` (a uuid v7 made when the call was recorded, also inside `payload`): one
  -- row per event, and a re-send after a crash between XADD and the `sent` update is deduplicated
  -- by billing.
  event_key       uuid        NOT NULL UNIQUE,
  -- the flat string→string field map billing's cost_flusher parses, frozen at record time
  payload         jsonb       NOT NULL,
  -- pending: not yet acknowledged by Redis. sent: XADD returned an id.
  status          text        NOT NULL DEFAULT 'pending',
  attempts        integer     NOT NULL DEFAULT 0,
  last_error      text,
  next_attempt_at timestamptz NOT NULL DEFAULT now(),
  sent_at         timestamptz,
  created_at      timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT ck_metering_outbox_status CHECK (status IN ('pending', 'sent')),
  CONSTRAINT ck_metering_outbox_sent_is_stamped CHECK (status <> 'sent' OR sent_at IS NOT NULL)
);

-- The drainer's claim: the oldest due pending rows. Partial: sent rows are history, not work.
CREATE INDEX metering_outbox_due_idx
  ON metering_outbox (next_attempt_at)
  WHERE status = 'pending';

-- The retention prune: sent rows older than the retention window.
CREATE INDEX metering_outbox_sent_idx
  ON metering_outbox (sent_at)
  WHERE status = 'sent';

COMMENT ON TABLE metering_outbox IS
  'Transactional outbox for the billing:costs stream (src/metering.rs). A pending row older than a '
  'few drain intervals is undelivered spend.';
