-- strata-rag schema v1 (spec §4) + invariant triggers.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE stores (
  id           uuid PRIMARY KEY,               -- uuid v7
  parent_id    uuid REFERENCES stores(id) ON DELETE RESTRICT,
  kind         text NOT NULL CHECK (kind IN ('router','leaf')),
  name         text NOT NULL,
  description  text NOT NULL,
  desc_vec     vector(1024) NOT NULL,          -- embedding of name + description
  content_vec  vector(1024),                   -- mean of contents (chunks, or children's content_vec)
  profile_vec  vector(1024) NOT NULL,          -- normalise(α·desc_vec + (1-α)·content_vec), α = 0.5
  token_count  bigint NOT NULL DEFAULT 0,      -- leaf: own; router: sum of descendants
  chunk_count  bigint NOT NULL DEFAULT 0,
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE NULLS NOT DISTINCT (parent_id, name)
);
-- exactly one root (parent_id IS NULL) is enforced in code; parent must be kind='router' (trigger).

CREATE TABLE documents (
  id            uuid PRIMARY KEY,
  store_id      uuid NOT NULL REFERENCES stores(id) ON DELETE CASCADE,  -- must be a leaf (trigger)
  source        text NOT NULL,                  -- file path or caller-supplied label
  title         text,
  content_hash  bytea NOT NULL,                 -- sha256 of normalised text
  token_count   integer NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (store_id, content_hash)
);

CREATE TABLE chunks (
  id           uuid PRIMARY KEY,
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  store_id     uuid NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
  ord          integer NOT NULL,
  text         text NOT NULL,
  token_count  integer NOT NULL,
  embedding    vector(1024) NOT NULL,
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
  UNIQUE (document_id, ord)
);
CREATE INDEX chunks_store_idx ON chunks (store_id);
CREATE INDEX chunks_tsv_idx   ON chunks USING gin (tsv);

-- ---------------------------------------------------------------------------
-- Invariant triggers. All use plain RAISE EXCEPTION (SQLSTATE P0001, raise_exception);
-- Task 8/9 map P0001 to StrataError::InvalidTarget. Do NOT add a custom ERRCODE.
-- A missing referenced row is left to the foreign key, which reports 23503.
-- ---------------------------------------------------------------------------

-- A store's parent must be a router.
CREATE FUNCTION stores_parent_must_be_router() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  parent_kind text;
BEGIN
  IF NEW.parent_id IS NULL THEN
    RETURN NEW;
  END IF;
  SELECT kind INTO parent_kind FROM stores WHERE id = NEW.parent_id;
  IF parent_kind IS NOT NULL AND parent_kind <> 'router' THEN
    RAISE EXCEPTION 'parent store % is a %; a parent must be a router', NEW.parent_id, parent_kind;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER stores_parent_must_be_router
  BEFORE INSERT OR UPDATE OF parent_id ON stores
  FOR EACH ROW EXECUTE FUNCTION stores_parent_must_be_router();

-- kind never changes. Without this, a router with children could become a leaf (or a leaf
-- holding documents a router) and the two kind-checking triggers would race.
CREATE FUNCTION stores_kind_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.kind <> OLD.kind THEN
    RAISE EXCEPTION 'store % kind is immutable (% -> %)', OLD.id, OLD.kind, NEW.kind;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER stores_kind_immutable
  BEFORE UPDATE OF kind ON stores
  FOR EACH ROW EXECUTE FUNCTION stores_kind_immutable();

-- Documents live in leaves and never move. Moving one would orphan its chunks' store_id.
CREATE FUNCTION documents_store_must_be_leaf() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  target_kind text;
BEGIN
  IF TG_OP = 'UPDATE' AND NEW.store_id IS DISTINCT FROM OLD.store_id THEN
    RAISE EXCEPTION 'document % cannot move from store % to %', OLD.id, OLD.store_id, NEW.store_id;
  END IF;
  SELECT kind INTO target_kind FROM stores WHERE id = NEW.store_id;
  IF target_kind IS NOT NULL AND target_kind <> 'leaf' THEN
    RAISE EXCEPTION 'store % is a %; documents must be in a leaf', NEW.store_id, target_kind;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER documents_store_must_be_leaf
  BEFORE INSERT OR UPDATE OF store_id ON documents
  FOR EACH ROW EXECUTE FUNCTION documents_store_must_be_leaf();

-- A chunk's store_id must equal its document's store_id (search filters on chunks.store_id).
CREATE FUNCTION chunks_store_matches_document() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  doc_store uuid;
BEGIN
  SELECT store_id INTO doc_store FROM documents WHERE id = NEW.document_id;
  IF doc_store IS NOT NULL AND doc_store <> NEW.store_id THEN
    RAISE EXCEPTION 'chunk store % does not match document % store %', NEW.store_id, NEW.document_id, doc_store;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER chunks_store_matches_document
  BEFORE INSERT OR UPDATE OF store_id, document_id ON chunks
  FOR EACH ROW EXECUTE FUNCTION chunks_store_matches_document();
