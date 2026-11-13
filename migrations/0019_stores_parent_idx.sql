-- Children-of-a-store lookups (`WHERE parent_id = $1`: child counts, the recursive subtree walks,
-- centroid recomputes) and the ON DELETE RESTRICT check of the parent_id FK had no index leading
-- with parent_id: the only one covering it is the (tenant, parent_id, name) unique key, which
-- leads with tenant. Not CONCURRENTLY: migrations run in a transaction, and stores is small.
CREATE INDEX IF NOT EXISTS stores_parent_id_idx ON stores (parent_id);
