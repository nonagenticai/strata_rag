-- Tenant scoping (spec §8.6): every store belongs to one tenant; documents and chunks belong to
-- their store's. A query, list, read or write only ever sees its own tenant's stores.
--
-- No default, deliberately: there is no tenant an existing store could be assigned to. On an empty
-- `stores` table (production at the time of writing) this succeeds; on a non-empty one
-- `ADD COLUMN ... NOT NULL` without a default FAILS the migration (23502, "contains null values")
-- instead of silently giving every existing store some made-up tenant. Assign tenants by hand (or
-- empty the table) before running it against data.
ALTER TABLE stores
  ADD COLUMN tenant uuid NOT NULL,
  ADD CONSTRAINT stores_tenant_not_nil CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid);

-- Names are unique among siblings within a tenant. The old (parent_id, name) key made root names
-- unique ACROSS tenants: a second tenant creating a root that another tenant already named got a
-- 409 duplicate_name, which leaks that the other tenant's store exists. A child's parent already
-- fixes its tenant, so only roots change meaning. This unique index also serves the
-- (tenant, parent_id) lookups (a tenant's roots, a store's children) as its leading columns.
ALTER TABLE stores DROP CONSTRAINT stores_parent_id_name_key;
ALTER TABLE stores
  ADD CONSTRAINT stores_tenant_parent_id_name_key UNIQUE NULLS NOT DISTINCT (tenant, parent_id, name);

-- A child store's tenant is its parent's. The store-insert transaction checks this too (with the
-- parent locked); the trigger holds it for every other writer. Plain RAISE EXCEPTION (P0001), as
-- the 0001 triggers.
CREATE FUNCTION stores_tenant_matches_parent() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  parent_tenant uuid;
BEGIN
  IF NEW.parent_id IS NULL THEN
    RETURN NEW;
  END IF;
  SELECT tenant INTO parent_tenant FROM stores WHERE id = NEW.parent_id;
  IF parent_tenant IS NOT NULL AND parent_tenant <> NEW.tenant THEN
    RAISE EXCEPTION 'store % tenant does not match its parent % tenant', NEW.id, NEW.parent_id;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER stores_tenant_matches_parent
  BEFORE INSERT OR UPDATE OF parent_id, tenant ON stores
  FOR EACH ROW EXECUTE FUNCTION stores_tenant_matches_parent();

-- A store never changes tenant: moving it would hand its documents to another tenant.
CREATE FUNCTION stores_tenant_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.tenant <> OLD.tenant THEN
    RAISE EXCEPTION 'store % tenant is immutable', OLD.id;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER stores_tenant_immutable
  BEFORE UPDATE OF tenant ON stores
  FOR EACH ROW EXECUTE FUNCTION stores_tenant_immutable();
