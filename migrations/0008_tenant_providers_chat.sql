-- Chat becomes a tenant-registered provider capability (spec §8.2): `chat` joins the capability
-- check, plus two chat-only request options. Both are NULL = "not sent", so a registration's
-- request stays plain OpenAI unless it opts in:
--   thinking    -> chat_template_kwargs.enable_thinking (a vLLM extension)
--   temperature -> temperature
ALTER TABLE tenant_providers DROP CONSTRAINT tenant_providers_capability;
ALTER TABLE tenant_providers
  ADD CONSTRAINT tenant_providers_capability CHECK (capability IN ('embed', 'rerank', 'chat'));
ALTER TABLE tenant_providers
  ADD COLUMN thinking bool,
  ADD COLUMN temperature double precision,
  ADD CONSTRAINT tenant_providers_chat_options_chat_only CHECK (
    capability = 'chat' OR (thinking IS NULL AND temperature IS NULL)
  );
