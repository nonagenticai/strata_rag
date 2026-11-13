-- A per-tenant generation of the store tree, bumped at COMMIT of every transaction that inserts,
-- updates or deletes a store row. Each strata-rag process caches a tenant's routing tree; that
-- cache checks this one indexed row on every query and reloads when it moved, so a write made by
-- ANOTHER process (a second replica, `strata reembed` run from the CLI) is seen by the next
-- query instead of only after a restart.
--
-- The trigger is DEFERRED to commit on purpose: every writer locks store rows child before parent
-- (see `db::stores`), and taking this row's lock in the middle of that walk would let two writers
-- deadlock (one holding the generation row, waiting for a store row the other holds). At commit
-- every store lock is already held, so this row is always the last lock taken.
CREATE TABLE store_generations (
  tenant     uuid   PRIMARY KEY,
  generation bigint NOT NULL
);

CREATE FUNCTION stores_bump_generation() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  t uuid;
BEGIN
  IF TG_OP = 'DELETE' THEN
    t := OLD.tenant;
  ELSE
    t := NEW.tenant;
  END IF;
  INSERT INTO store_generations (tenant, generation) VALUES (t, 1)
  ON CONFLICT (tenant) DO UPDATE SET generation = store_generations.generation + 1;
  RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER stores_bump_generation
  AFTER INSERT OR UPDATE OR DELETE ON stores
  DEFERRABLE INITIALLY DEFERRED
  FOR EACH ROW EXECUTE FUNCTION stores_bump_generation();
