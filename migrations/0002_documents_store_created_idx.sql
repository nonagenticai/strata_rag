-- Keyset pagination for GET /api/stores/{id}/documents: WHERE store_id = $1
-- AND (created_at, id) < (...) ORDER BY created_at DESC, id DESC.
CREATE INDEX documents_store_created_idx ON documents (store_id, created_at DESC, id DESC);
