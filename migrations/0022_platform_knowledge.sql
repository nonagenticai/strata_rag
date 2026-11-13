-- Platform knowledge (docs/superpowers/specs/2026-10-20-platform-knowledge-design.md §3, §4, §6).
--
-- `platform_packs`: a pack version is a root store owned by the platform owner
-- (`STRATA_PLATFORM_OWNER`; the database does not know that uuid, so the app checks the owner).
-- Draft while `published_at IS NULL`; published = frozen (`db::platform::assert_writable`);
-- retired = hidden from listings. A pack row holds its root store: ON DELETE RESTRICT, the app
-- removes the row of a draft or retired pack in the store-delete transaction.
--
-- `platform_exclusions`: a tenant excluded from `key_prefix` and every key below it, segment-wise.
--
-- `documents.topic_keys` / `documents.supersedes`: the shadowing sets (§6.1), set at ingest.
--
-- No row writes here, so the 0010 / 0020 ALTER trap (README) does not apply.

CREATE TABLE platform_packs (
  key          text        NOT NULL,
  version      int         NOT NULL,
  root_id      uuid        NOT NULL UNIQUE REFERENCES stores(id) ON DELETE RESTRICT,
  published_at timestamptz,
  retired_at   timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (key, version),
  -- 1 to 6 `/`-separated segments of [a-z0-9_-]
  CONSTRAINT platform_packs_key CHECK (key ~ '^[a-z0-9_-]+(/[a-z0-9_-]+){0,5}$'),
  CONSTRAINT platform_packs_version CHECK (version >= 1),
  CONSTRAINT platform_packs_retired_published
    CHECK (retired_at IS NULL OR published_at IS NOT NULL)
);

-- A pack's root is a root store (no parent). Plain RAISE EXCEPTION (P0001), as the other triggers.
CREATE FUNCTION platform_packs_root_is_root() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM stores WHERE id = NEW.root_id AND parent_id IS NULL) THEN
    RAISE EXCEPTION 'store % is not a root store (it has a parent or does not exist)', NEW.root_id;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER platform_packs_root_is_root
  BEFORE INSERT OR UPDATE OF root_id ON platform_packs
  FOR EACH ROW EXECUTE FUNCTION platform_packs_root_is_root();

CREATE TABLE platform_exclusions (
  tenant     uuid        NOT NULL,
  key_prefix text        NOT NULL,
  reason     text        NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant, key_prefix),
  CONSTRAINT platform_exclusions_tenant_not_nil
    CHECK (tenant <> '00000000-0000-0000-0000-000000000000'::uuid),
  CONSTRAINT platform_exclusions_key_prefix
    CHECK (key_prefix ~ '^[a-z0-9_-]+(/[a-z0-9_-]+){0,5}$')
);

-- Constant defaults: metadata-only, no table rewrite.
ALTER TABLE documents
  ADD COLUMN topic_keys text[] NOT NULL DEFAULT '{}',
  ADD COLUMN supersedes text[] NOT NULL DEFAULT '{}';

CREATE INDEX documents_topic_keys_gin ON documents USING gin (topic_keys);
CREATE INDEX documents_supersedes_gin ON documents USING gin (supersedes);
