-- Chat becomes a per-store capability too (agent D92): a store may bind its own chat provider,
-- resolved like embed / rerank: the store's own row -> the nearest ancestor's -> the tenant's
-- (`tenant_providers`) -> the dev default -> 409 provider_not_configured. Same chat-only request
-- options as the tenant table (0008), NULL = "not sent". The key's associated data stays
-- tenant || store || capability (0016).
--
-- Drift-tolerant: every step can run again on a database that already has it. No `stores` write,
-- so the 0010 trap does not apply.
ALTER TABLE store_providers DROP CONSTRAINT IF EXISTS store_providers_capability;
ALTER TABLE store_providers
  ADD CONSTRAINT store_providers_capability CHECK (capability IN ('embed', 'rerank', 'chat'));
ALTER TABLE store_providers
  ADD COLUMN IF NOT EXISTS thinking bool,
  ADD COLUMN IF NOT EXISTS temperature double precision;
ALTER TABLE store_providers DROP CONSTRAINT IF EXISTS store_providers_chat_options_chat_only;
ALTER TABLE store_providers
  ADD CONSTRAINT store_providers_chat_options_chat_only CHECK (
    capability = 'chat' OR (thinking IS NULL AND temperature IS NULL)
  );
