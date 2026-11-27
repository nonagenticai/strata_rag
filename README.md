# strata-rag

[![CI](https://github.com/nonagenticai/strata_rag/actions/workflows/ci-public.yml/badge.svg)](https://github.com/nonagenticai/strata_rag/actions/workflows/ci-public.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/strata_rag)](https://github.com/nonagenticai/strata_rag/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Multi-layer routed RAG: a tree of router/leaf stores on Postgres + pgvector. strata runs no model:
embeddings (bge-m3, 1024 dims) and reranking come from HTTP providers, so it is a CPU-only service.
Every store belongs to one tenant; a query routes down the tree from a root through router stores to
leaves, and a store may bind its own embed, rerank and chat providers.

| | |
|---|---|
| **Runtime** | Rust — edition 2024, toolchain 1.100.0 (`Cargo.toml`, `rust-toolchain.toml`) |
| **Depends on** | PostgreSQL 18 + pgvector (`docker-compose.yml`); an embed provider (OpenAI dialect) and a rerank provider (Cohere v2) over HTTP; optional Redis for metering; optional chat provider for `/api/ask` |
| **Runs on** | Docker (CI publishes to GHCR) or Kubernetes; a native binary for local development; API on port 8092 |

## Quick start

```bash
git clone https://github.com/nonagenticai/strata_rag.git
cd strata_rag
docker compose up -d            # PostgreSQL 18 + pgvector on 127.0.0.1:5440
cargo build --locked
```

`strata serve` refuses to start without a `providers.toml`; a minimal dev one (replace the base
URLs and the key with a real provider):

```toml
# providers.toml
region = "eu"
mode = "dev"
[policy]
allowed_dialects = ["openai", "cohere"]

[default.embed]
base_url = "https://api.example.eu/v1"
dialect = "openai"
model = "BAAI/bge-m3"
dims = 1024
api_key = "env:STRATA_DEV_API_KEY"
regions = ["eu"]
zdr = true

[default.rerank]
base_url = "https://api.example.eu/v2"
dialect = "cohere"
model = "BAAI/bge-reranker-v2-m3"
api_key = "env:STRATA_DEV_API_KEY"
regions = ["eu"]
zdr = true
```

```bash
export STRATA_DEV_API_KEY=...                     # your provider key
STRATA_PROVIDERS_FILE=$PWD/providers.toml ./target/debug/strata serve
```

The database URL defaults to the `docker compose` container, so this local run needs no
`STRATA_DATABASE_URL`. The API listens on `127.0.0.1:8092` (`STRATA_BIND`).

## Usage

Health is the only unauthenticated route; with the dev providers above reachable it answers:

```bash
curl -s http://127.0.0.1:8092/health
{"db":"ok","embed":"ok","rerank":"ok","chat":"unconfigured","laya":"unconfigured"}
```

Each component is `ok`, `down` (with an `*_error` class), `busy`, or `unconfigured`. The CLI runs
the same engine directly against the database:

```bash
TENANT=0190f1b2-0000-7000-8000-000000000001
./target/debug/strata store add --tenant "$TENANT" --kind leaf --name notes \
  --description "Personal notes"
# created leaf notes <store-uuid>
./target/debug/strata ingest --tenant "$TENANT" notes notes.txt
./target/debug/strata query --tenant "$TENANT" "what did I write about ports?"
```

`strata store ls` prints the tree; `strata query --json` prints the full `QueryResponse`;
`strata ask` generates an answer when a chat provider is configured.

## Deploy

strata-rag runs as a container (below). A native binary is for local development.

| | Local (dev) | Docker |
|---|---|---|
| Build | `cargo build --release` | `docker build .` |
| How | `strata serve` with the variables below in the environment | `Dockerfile` |
| Tokenizer | `~/.cache/huggingface` (bge-m3 `tokenizer.json` only, no weights, fetched on first start) | baked into the image at `/usr/share/strata/tokenizer.json` (`STRATA_TOKENIZER_PATH`), sha256-pinned |
| Port | 8092 | 8092 (`EXPOSE 8092`, same port in and out) |

Never use port 8000.

**Cluster:** CI builds and publishes `ghcr.io/<owner>/strata-rag`; the deployment manifests live in
the operator's own gitops repository, not here.

## Configuration

### Providers

Embed, rerank and chat providers come from two layers (spec §4.1, §8.2):

1. **Tenant registrations** — each tenant registers its own providers with
   `PUT /api/tenant/providers` (below); stored in `tenant_providers`, keys encrypted at rest.
2. **Deployment config** — `providers.toml` (`STRATA_PROVIDERS_FILE`, default
   `/etc/strata/providers.toml`): the deployment `region`, `mode`, the dialect policy and, in
   `mode = "dev"` only, default providers. A production deployment has no defaults: a tenant without
   a registration gets `409 provider_not_configured` (naming the capability) on the first call that
   needs it.

```toml
region = "eu"                 # this deployment's region: "eu" | "us"
mode = "dev"                  # "dev" | "production" (production: no [default.*] allowed)

[policy]
allowed_dialects = ["openai", "cohere"]   # embed = openai, rerank = cohere (the only implemented ones)
allow_internal_provider_hosts = false     # dev only (true in production = startup error)

[default.embed]               # dev only
base_url = "http://127.0.0.1:9131/v1"
dialect = "openai"
model = "BAAI/bge-m3"
# model_id = "BAAI/bge-m3@1024"   # store identity; default "{model}@{dims}"
dims = 1024                   # must be 1024
api_key = "env:MLX_API_KEY"   # env references only; a literal key is a startup error
regions = ["eu"]              # must contain `region`
zdr = true                    # required in an eu deployment
timeout_ms = 30000            # default 30000 (embed) / 10000 (rerank)
concurrency = 8               # concurrent query-path calls to this provider

[default.rerank]              # dev only
base_url = "http://127.0.0.1:9131/v2"
dialect = "cohere"
model = "BAAI/bge-reranker-v2-m3"
api_key = "env:MLX_API_KEY"
regions = ["eu"]
zdr = true

[default.chat]                # dev only, OPTIONAL: /api/ask and query expansion (HyDE)
base_url = "https://api.nonagentic.ai/vllm/v1"   # POST {base_url}/chat/completions (OpenAI)
dialect = "openai"
model = "local-llm"
api_key = "env:VLLM_LICENSE_TOKEN"   # sent as Authorization: Bearer <licence>
regions = ["eu"]
zdr = true
timeout_ms = 15000            # behind care-bus (faq_ask gives up at 18 s); default 60000
thinking = false              # chat_template_kwargs.enable_thinking (vLLM); on, the model can
                              # spend the whole max_tokens budget reasoning; omit = not sent
temperature = 0.2             # omit = not sent (the server's default)
routed = false                # chat only (refused on embed/rerank; tenants cannot set it): true =
                              # base_url is the inference-router, which bills the call: strata sends
                              # X-Effective-Tenant-Id: <tenant>, X-Project-Id (the caller's, else
                              # tenant-level) and the caller's X-App-Id / X-Actor-Id (relayed,
                              # omitted when not sent), and emits no strata-rag-chat event
```

**Chat** (`dialect = "openai"`, no `dims`; optional `thinking` / `temperature`, each sent only
when set, so the request is plain OpenAI `{model, messages, max_tokens}` otherwise; `content` may
come back as a string or an array of text parts): without a chat provider (no
registration, no `[default.chat]`) `/api/ask` and `expand: true` queries answer
`409 provider_not_configured` naming `chat`; plain queries are unaffected. A configured provider
that fails makes `/api/ask` a `503 generation_unavailable` (the message carries the error class)
and degrades expansion to the question alone (`expansion.fallback`), as does an expansion that
takes longer than 5 s. Chat calls are never retried. A tenant chat `timeout_ms` is at most 60000;
the registration probe is one completion with `max_tokens = 1`, bounded by `min(timeout_ms, 30 s)`;
responses are capped at 1 MiB. A stored registration that cannot be used (a refused host, a key
that does not decrypt) fails only the requests that need that capability.

Defaults are validated at startup with the same rules as a registration (except the live probe);
any violation exits non-zero.

| Call | Shape | Body |
|---|---|---|
| embed | OpenAI `POST {base_url}/embeddings` | `{model, input, dimensions: 1024, encoding_format: "float", input_type: "query"\|"document"}` |
| rerank | Cohere v2 `POST {base_url}/rerank` | `{model, query, documents, top_n}` |

`Authorization: Bearer <key>` is sent only when a key is set. 401/403 are `auth`, 408/429/5xx
`retryable` (retried twice with a short backoff; a client timeout is not retried), other 4xx
`invalid_request` (never retried); all surface as 502 `upstream_unavailable`. Each store records the
identity of the model that embedded it (the resolved embed provider's `model_id`, default
`{model}@{dims}`). Querying a tree, ingesting into a leaf, or creating or re-describing a store
answers 409 `embed_model_mismatch` when that store or any of its ancestors (or, for a query, any
store under the start) holds a different identity — so changing a tenant's embed model makes its
existing stores 409 until they are re-embedded.

#### Tenant provider API

`PUT /api/tenant/providers` (same auth and tenant rules as every route) replaces the tenant's set:

```json
{
  "embed":  {"base_url": "https://api.example.eu/v1", "dialect": "openai", "model": "acme-embed",
             "dims": 1024, "regions": ["eu"], "zdr": true, "api_key": "sk-...",
             "model_id": null, "timeout_ms": 30000, "concurrency": 8},
  "rerank": {"base_url": "https://api.example.eu/v2", "dialect": "cohere", "model": "acme-rerank",
             "regions": ["eu"], "zdr": true, "api_key": "sk-..."},
  "chat":   {"base_url": "https://api.example.eu/v1", "dialect": "openai", "model": "acme-chat",
             "regions": ["eu"], "zdr": true, "api_key": "sk-...", "thinking": false,
             "temperature": 0.2}
}
```

- A capability that is `null` or left out is removed.
- `api_key` is write-only. Omitted: the stored key is kept when `base_url` is unchanged; with a new
  `base_url` and a stored key it is `422 api_key_required` (a stored key never goes to a new host).
- Validation, each a `422` with its own `error.code`: `region_not_allowed` (regions must contain the
  deployment region), `zdr_required` (eu), `dims_must_be_1024`, `dialect_not_allowed`,
  `invalid_provider` (URL, model, numbers), `probe_failed` (a live call: one 1-text embed that must
  come back 1024-d and already normalised; one 2-document rerank with valid indices and scores in
  [0, 1]; ≤ 10 s; the message names the provider error class only, never the key or URL).
- Needs `STRATA_PROVIDER_KEY`; without it the PUT is `503 provider_key_unconfigured`.
- `base_url` is parsed with `url::Url` (reqwest's parser) and accepted only in canonical form:
  no backslash, whitespace or control characters, no userinfo, query or fragment, and
  `url.as_str()` equal to the input, so the host checked is the host fetched.
- SSRF guard (unless the dev-only `allow_internal_provider_hosts = true`): `base_url` must be
  `https`; its host must not be an IP or name in a non-public range (loopback, 10/8, 172.16/12,
  192.168/16, 169.254/16, 100.64/10, documentation / benchmarking ranges, fc00::/7, fe80::/10,
  2001:db8::/32, Teredo, 6to4 of a blocked IPv4, unspecified, multicast, broadcast), nor a
  single-label name or end in `.local` / `.internal` / `.cluster.local`, and every address it
  resolves to must be public — else `422 provider_host_not_allowed`. Tenant clients also resolve
  through a filtering resolver (no proxy, no redirects), so a name that later re-resolves to an
  internal address fails to connect. `providers.toml` defaults are operator-set and exempt.

`GET /api/tenant/providers` (and the PUT's answer) shows per capability `source`
(`tenant` | `default` | `unconfigured`); a registration adds `base_url, dialect, model, model_id,
dims, regions, zdr, timeout_ms, concurrency, has_api_key, updated_at`, a dev default only
`dialect, model, dims` (never where it runs). No read returns a key.

`POST /api/tenant/providers/test` tries one provider without saving it: body = a registration plus
`capability` (`embed` | `rerank` | `chat`); minimal `{capability, dialect, base_url, model,
api_key?}` (`regions`, `zdr`, embed `dims` default to what the deployment accepts — the save still
enforces them). Same validation, SSRF guard and live probe as the PUT; nothing is stored, cached or
metered (no cost event: a probe is not a tenant's use). `api_key` omitted = the tenant's stored key
of that capability when `base_url` is unchanged, else `422 secret_required_for_new_host`. Answers
200 `{ok, capability, model, latency_ms, dims?, error_class?, detail}` for any provider answer (a
failing provider is `ok: false` with its error class, never a 5xx); 422 for an invalid body or a
refused host. No answer carries the key or the URL.

A tenant's `timeout_ms` is at most 30000 (embed) / 15000 (rerank). Provider responses are read up
to a cap (embed: texts × 1024 × 24 bytes + 64 KiB; rerank: 1 MiB); a longer body is an
`invalid_response` and is never parsed.

Resolution per request: the tenant's registration, else the dev default, else 409. Built clients are
cached per (tenant, capability) and checked against the row's `updated_at` on every request, so a PUT
on any replica takes effect on the next request everywhere. Each resolved provider keeps its own
query slots (`concurrency`) and ingest slots (`STRATA_PROVIDER_BULK_CONCURRENCY`, capped at
`concurrency`).

#### Store providers and re-embed copy

A store may bind its own embed / rerank / chat provider (`store_providers`, migrations 0016 and
0017; chat per store since D92). Every request acting on a store (create a child, re-describe,
ingest, query or ask from it as start, `strata reembed`) resolves per capability: the store's own registration → the
nearest ancestor's → the tenant's → the dev default → 409 `provider_not_configured`.

- `GET /api/stores/{id}/providers`: `{store_id, embed_model, embed, rerank, chat}`, each capability with
  `source` (`store` | `ancestor` + `from_store` | `tenant` | `default` | `unconfigured`) and the
  settings as in the tenant view. Never a key.
- `PUT /api/stores/{id}/providers` `{embed?, rerank?, chat?}`: the same body, validation, live
  probe, SSRF guard and key rules as the tenant PUT (chat with its optional `thinking` /
  `temperature`; any other field is `422 invalid_input`); a capability left
  out is removed (inherited again). An `embed` whose model identity differs from the store's
  `embed_model` is `409 embed_model_mismatch`, unless the store is empty (no documents, no child
  stores) and its ancestors already hold that model: the store is then re-tagged with the new
  model and its description embedded again by the new provider.
- `DELETE /api/stores/{id}/providers`: removes the store's own registrations.
- `POST /api/stores/{id}/reembed-copy` `{"target_store": "<uuid>"}` → `202` with the job
  `{job, source_store, target_store, embed_model, state: "running", chunks_done, chunks_total,
  started_at}`. A background task re-embeds every document of `id`'s subtree into the target with
  the target's providers: the descendants are mirrored under the target (kind, name, description),
  documents are copied with their provenance, every chunk text is embedded again, all in one
  transaction at the end. The target must be another store of the same kind, empty, outside the
  source's subtree (`422 invalid_target`) and in its resolved model (`409 embed_model_mismatch`);
  one copy per target at a time (`409 conflict`). The source is not changed, and keeps taking
  writes (D91): after the snapshot pass, delta passes embed what the source gained meanwhile
  (new documents, new or renamed stores) until the final transaction, holding the target and the
  source subtree locked, finds nothing new; it then writes the copy and marks the job `done`.
  A document deleted from the source before that is not copied. At most 8 passes; a source that
  never holds still fails the job `conflict: re-embed copy did not converge ...` with nothing
  written. Only a request starts a copy. With `"copy_vectors": true` (default false) the copy
  reuses the source's vectors (chunk embeddings, store description vectors; delta passes too)
  instead of embedding again: no provider call, for a rerank- or chat-only change. It needs the
  target to resolve to the source's embed model, else `409 embed_model_mismatch` (no job). The
  job carries `copy_vectors` (migration 0018).
- `GET /api/stores/{id}/reembed-copy/{job}` → the job; `state` `running` | `done` | `failed`
  (`error`). A running job whose process stopped beating for 60 s reads `failed` (`interrupted`).

The agent's flow to move a store to another model: create an empty root, `PUT` its providers with
the new embed (re-tag), `POST .../reembed-copy` from the old root into it, poll the job, then query
with `root_id` = the new root.

Keys at rest: AES-256-GCM under `STRATA_PROVIDER_KEY` (base64 of 32 random bytes:
`openssl rand -base64 32`), a random 96-bit nonce per write, `tenant || capability` (a store
row: `tenant || store || capability`) as associated data (a ciphertext moved to another row does
not decrypt). A process finding stored keys without
`STRATA_PROVIDER_KEY` refuses to start. `/health` probes the dev defaults only (`unconfigured` in
production), never a tenant's registration.

### Environment

| Variable | Notes |
|---|---|
| `STRATA_DATABASE_URL` | Postgres with pgvector. Default `postgres://strata_rag:strata_rag@127.0.0.1:5440/strata_rag`, which is the local `strata-rag-db` container from `docker-compose.yml`. A production DB follows one service, one DB, one user (`strata_rag` on a Postgres you provision; the pgvector extension must be available there). |
| `STRATA_DB_MAX_CONNECTIONS` / `STRATA_DB_IDLE_TIMEOUT_S` / `STRATA_DB_MAX_LIFETIME_S` | Postgres pool: connections per process (default 16; the pgbouncer `pool_size` must cover this × (replicas + surge)), seconds an idle connection is kept (default 120; keep it below pgbouncer `server_idle_timeout` 240 and postgres `idle_session_timeout` 600, or the server closes it first and the next query fails with EOF), seconds before any connection is recycled (default 1800, ≥ the idle timeout). |
| `STRATA_EXACT_SCAN_MAX_CHUNKS` | Leaf sets of at most this many chunks (the sum of their `stores.content_chunks`) are searched exactly instead of through the HNSW index (default 5000, 0–1000000; 0 = always HNSW). See [Search index](#search-index). |
| `STRATA_TOKENIZER_PATH` | Local bge-m3 `tokenizer.json` to use instead of the hf-hub cache/download. Unset (native default) = hf-hub; set in the image. A missing or unreadable file fails startup naming this variable. |
| `STRATA_BIND` | Default `127.0.0.1:8092`. Use the host's LAN IP for access from other hosts. |
| `STRATA_METRICS_BIND` | Optional second listener serving only `GET /metrics` (no auth, no API routes; every other path is 404), e.g. `0.0.0.0:8093`, so Prometheus can scrape without reaching the (mTLS) API port. Unset or empty (default) = disabled; must differ from `STRATA_BIND`. `/metrics` stays on the API port too. |
| `STRATA_API_TOKENS` | Comma-separated `<tenant-uuid>:<token>` (tenant-pinned) or `*:<token>` (multi-tenant service caller) entries; see [Tenants](#tenants). Required when `STRATA_BIND` is not loopback: the service refuses to start without them. A bare token, a nil or invalid uuid, or a repeated token is a startup error. |
| `STRATA_LICENSE_INTROSPECT_URL` | Licence-token auth: the licences service's full `POST /license/introspect` URL. Set = a bearer matching no `STRATA_API_TOKENS` entry is introspected there and accepted when the licence is active and its app key is in `STRATA_LICENSE_APP_KEYS`; the caller acts for the licence's tenant (see [Tenants](#tenants)). An unknown or inactive licence is `401`; an active licence of an app not in that list is `403 licence_app_not_allowed`; the licences service unreachable or answering non-2xx = `503 licence_unavailable` (fail closed). Also satisfies the non-loopback auth requirement. Unset = off (static tokens only). |
| `STRATA_LICENSE_APP_KEYS` | Comma-separated app keys whose licences are accepted. Default `strata-rag`. |
| `STRATA_LICENSE_CACHE_S` / `STRATA_LICENSE_TIMEOUT_MS` | Seconds an accepted licence token is reused without re-introspecting (default 30, 0–60; 0 = every request; cached by token SHA-256, at most 4096 entries; rejections are never cached), so a revoked licence stops working within this window / timeout of one introspect call (default 3000). |
| `STRATA_TENANT` | CLI only: the tenant uuid every engine command acts for (same as `--tenant`). |
| `STRATA_LICENSES_FILE` | File holding `LAYA_LICENSE_TOKEN` / `VLLM_LICENSE_TOKEN` (default `~/.config/strata-rag/licenses.env`, mode 600). |
| `STRATA_PROVIDERS_FILE` | The deployment's `providers.toml` (see [Providers](#providers)). Default `/etc/strata/providers.toml`. Missing or invalid: startup fails. The flat `STRATA_EMBED_*` / `STRATA_RERANK_*` variables are gone. |
| `STRATA_PROVIDER_KEY` | Base64 of the 32-byte key sealing tenant provider keys. Secret. Required for `PUT /api/tenant/providers` and whenever a stored key exists. |
| `STRATA_BULK_EMBED_BATCH` / `STRATA_PROVIDER_BULK_CONCURRENCY` | Texts per ingest embed request (default 32) / concurrent ingest embed requests per embed provider (default 2). |
| `STRATA_HNSW_EF_SEARCH` | `hnsw.ef_search` for the dense search (default 100, 1–1000). See [Search index](#search-index). |
| `STRATA_QUERY_EMBED_CACHE` | Query vectors kept in the LRU (default 1024, ~4 KiB each; 0 = off). See [Query-embedding cache](#query-embedding-cache). |
| `METERING_REDIS_URL` | billing's Redis (`redis://...`), and the metering switch: set = every process records cost events in `metering_outbox`, `strata serve` drains them to this Redis. Secret (may carry a password). Unset = no cost events. See [Metering](#metering). |
| `STRATA_PLATFORM_OWNER` / `STRATA_PLATFORM_LICENSES` / `STRATA_PLATFORM_PENALTY` | Platform knowledge: see [Tenants](#tenants) and [Platform knowledge](#platform-knowledge). |
| `METERING_STREAM` | Stream the cost events go to. Default `billing:costs`; set it explicitly in the deployment (billing's emitter guide). |

Other tuning knobs are in `.env.example`.

## Reference

### Docker

Base images are `debian:12-slim` from Docker Hub. There is no glibc Rust image, so the builder is
`debian:12-slim` with Rust installed through rustup, and the runtime is `debian:12-slim`. The image runs as uid 65532 and has a `HEALTHCHECK` on
`/health`. `/health` returns 200 with a per-component JSON body (`db`, `embed`, `rerank`, `chat`,
`laya`; `embed` / `rerank` / `chat` are the dev default providers, `unconfigured` in production or
without that default — `chat` replaced the old `vllm` component and checks `GET {base_url}/models`;
a down provider adds `embed_error` / `rerank_error` / `chat_error` with its error class), so the healthcheck shows the
server is up, not that its dependencies are.

```sh
docker build -t strata-rag .
docker run -d --name strata-rag -p 127.0.0.1:8092:8092 --read-only \
  -e STRATA_DATABASE_URL=postgres://... -e STRATA_API_TOKENS=... \
  -e STRATA_PROVIDER_KEY=... -v "$PWD/providers.toml:/etc/strata/providers.toml:ro" strata-rag
```

### Consumers

care-bus reaches the service through `STRATA_RAG_URL`, the base URL without a trailing path. Use
`http://127.0.0.1:8092` on the same host, or `http://<host-ip>:8092` from another host,
and send one of the `STRATA_API_TOKENS` as a bearer token.

### Tenants

Every store belongs to one tenant (a uuid); its documents and chunks belong to it too. Every
route but `/health` acts for exactly one tenant and never reads or changes another tenant's
data. Another tenant's store or document id answers `404 not_found`, the same as an id that does
not exist.

| caller | tenant |
|---|---|
| `<tenant-uuid>:<token>` | that tenant. `X-Tenant-Id` is optional; if sent, it must name the same tenant (otherwise `403 tenant_mismatch`). |
| `*:<token>` | `X-Tenant-Id: <tenant-uuid>` is required (otherwise `400 tenant_required`). |
| licence token (`STRATA_LICENSE_INTROSPECT_URL` set) | the licence's `tenant_id`. `X-Tenant-Id` is optional; if sent, it must name the same tenant (otherwise `403 tenant_mismatch`). |
| no tokens configured (loopback only) | `X-Tenant-Id` is required, as with `*`. |

`X-Tenant-Id` takes the bare tenant uuid. It does not take a licence token or a `Bearer ...`
value: those, a malformed uuid, and the nil uuid answer `400 invalid_tenant`. A missing or wrong
bearer token is `401` first.

A tenant may hold several root stores (one tree per embedding model, see "Store providers and
re-embed copy"); names are unique among siblings within a tenant, roots included. `GET
/api/stores` lists the caller's stores only. `/api/query` and `/api/ask` route from `root_id`
(which must be one of the caller's stores, else 404), or without it from the caller's only root;
a caller with more than one root and no `root_id` gets `409 ambiguous_root` (D90), as does a
tenant `/api/eval`. `DELETE
/api/sources/{source_id}/documents` prunes only the caller's documents of that source.
`/api/eval` scores the caller's stores.

The CLI takes `--tenant <uuid>` (or `STRATA_TENANT`) on every command except `serve`.

**Platform owner.** `STRATA_PLATFORM_OWNER` (a non-nil uuid; invalid = startup error) names the
tenant that owns [platform knowledge](#platform-knowledge). Its stores are ordinary stores of that
tenant; the curator writes with an ordinary `<platform-uuid>:<token>` entry. Unset = platform
knowledge off: every `/api/platform/*` route is 404 and queries behave as before.
`STRATA_PLATFORM_LICENSES` (comma list, default
`CC0-1.0,PDDL-1.0,public-domain,CC-BY-4.0,CC-BY-SA-4.0,OGL-UK-3.0,ODC-BY-1.0`) is the licence
allow-list for platform ingest; `STRATA_PLATFORM_PENALTY` (default 0.05, 0–1) is the default
platform score penalty. The only cross-tenant read is a published pack (below).

### Query filters

`/api/query` and `/api/ask` take optional document filters. Both are applied in SQL during the
candidate search, so a filtered query still returns `top_k` hits whenever enough documents match:

| field | rule |
|---|---|
| `source_ids` | non-empty list (at most 1000): only chunks of documents whose `source_id` is one of them |
| `metadata` | JSON object: only chunks of documents whose `metadata` contains it (`@>`), e.g. `{"lang":"en"}` |

Platform precedence fields (see [Platform knowledge](#platform-knowledge)):

| field | rule |
|---|---|
| `supersedes` | topic keys added to the shadow set: platform hits whose document `topic_keys` overlap it are dropped before rerank |
| `platform_penalty` | subtracted from each platform hit's reranked score (default `STRATA_PLATFORM_PENALTY`) |
| `platform_min_score` | platform hits below it (after the penalty) are dropped (default none) |

Every result carries `layer` (`"tenant"` / `"platform"`) and, for platform hits, `pack`
(`"<key>@<version>"`); `/api/ask` labels each source `(tenant: …)` / `(platform <key>@<v>: …)`.

CLI: `strata query --tenant <uuid> "q" --source-id a --source-id b --metadata '{"lang":"en"}'`.

### Document sources and re-sync

Each document can carry its caller-side provenance. Send these fields in the JSON ingest body of
`POST /api/stores/{id}/documents` (`source_id` and `source_run` also work as query params with a
`text/plain` body):

| field | rule |
|---|---|
| `source_id` | the source the document came from (for example a care-bus source). Trimmed, 1–200 chars |
| `source_run` | the sync run that delivered it. Same limits; only allowed together with `source_id` (422 otherwise) |
| `metadata` | the source's mapped fields. Must be a JSON object (422 otherwise), at most 64 KiB as JSON (413). Absent means `{}` |
| `topic_keys` | JSON only. Topic keys the document covers (platform documents), at most 32 keys of 128 chars; stored sorted, deduplicated |
| `supersedes` | JSON only. Topic keys the document overrides (tenant documents shadowing platform ones); same limits |
| `provenance` | JSON only. `{source_url, license, retrieved_at}` (non-blank, `retrieved_at` RFC 3339), stored as `metadata.provenance`. Required in a platform store, with a licence on `STRATA_PLATFORM_LICENSES` (422 otherwise) |

The three source fields come back in the ingest response, in each item of
`GET /api/stores/{id}/documents` (filter with `?source_id=`, which works with `cursor`), and in
each `results[]` hit of `/api/query` and `/api/ask`. `topic_keys` and `supersedes` come back in
the ingest response and in each item of `GET /api/stores/{id}/documents`, always as arrays (`[]`
when unset).

If identical text is ingested again into the same store under the **same** `source_id`, nothing
is re-embedded. The existing document takes the request's `source` label. It also takes
`source_run`, `metadata` (replaced whole, not merged) and `title` when the request sends them; a
field the request leaves out keeps its stored value, so a re-delivery never erases a run. The
answer is `200 {"status":"existing",...}`.

**Cross-source dedupe:** a store keeps one copy of any given content. If the text is already
stored there under a **different** source (or under no source), the answer is that existing
document, unchanged, with **its own** `source_id`, `source_run` and `metadata`. It is not re-tagged
to the new source. Callers should compare the `source_id` they get back with the one they sent. A
mismatch means another source owns the content, and pruning that owning source deletes the
document.

Re-sync recipe for a source `S`:

1. Pick a new run id `R`, then ingest every current document of `S` with `source_id=S`,
   `source_run=R`. Unchanged documents are only re-tagged; new or changed ones are embedded.
2. `DELETE /api/sources/S/documents?except_run=R` deletes the documents of `S` that do not carry
   `R` (documents that were removed or changed at the source). The answer is
   `200 {"deleted":n,"kept":m,"stores":[leaf ids],"refused_stores":[]}`.

The old documents stay searchable throughout. The prune runs one locked transaction per leaf,
and each leaf has its own empty-sync guard. If a leaf holds documents of `S` but none of them
carries `R`, that leaf is left untouched and listed in `refused_stores`, because its sync most
likely failed. When any leaf is refused, the answer is `409 empty_sync_run`. Its body still
carries `deleted`, `kept`, `stores` and `refused_stores` next to `error`, because other leaves
may have been pruned. `allow_empty=true` turns the guard off. `store_id=<leaf uuid>` limits the
prune to one leaf. To delete every document of a source, send `all=true` instead of `except_run`.
Exactly one of the two is required, otherwise the answer is 422. An unknown source answers
`{"deleted":0,...}`.

CLI equivalents:
`strata ingest <leaf> <path> --source-id S --source-run R --metadata '{"url":"..."}'`
and `strata source prune S (--except-run R | --all) [--store <leaf>] [--allow-empty]`.

### Platform knowledge

Platform knowledge (spec `docs/superpowers/specs/2026-10-20-platform-knowledge-design.md`) is
knowledge every tenant may read, owned by `STRATA_PLATFORM_OWNER`. A **pack** is a numbered,
immutable version `key@version` of one platform root store (`key`: 1–6 `/`-separated segments
of `[a-z0-9_-]`, e.g. `industry/retail`).

- **Draft**: visible and writable to the platform owner only. **Published**: frozen: any write
  into the pack's subtree (ingest, document or store delete, store create/update, reembed,
  reembed-copy target, provider binding) is `409 pack_published`. A new version is a new root
  registered as `version + 1`; callers pin `key@version` and strata never moves a pin.
- **Retired**: hidden from other tenants' listing, still queryable by `root_id` for existing pins;
  the curator may then delete its root (which drops the pack row; a draft's root likewise).
- **Exclusions**: a tenant excluded from a prefix (`industry/retail`) loses that key and every
  key below it, segment-wise, in all versions.
- **Reads**: a default query searches only the caller's own tree; platform content is searched
  when the caller names a readable pack's stores in `root_ids` / `leaf_ids` (leaf ids from
  `GET /api/platform/packs/{key}/{version}/stores`).
- **Precedence (tenant wins)**: shadowing by `supersedes` (the caller's searched documents plus the
  request), then `platform_penalty` / `platform_min_score` after rerank, then dropping a platform
  hit that near-duplicates a kept tenant hit (word-5-shingle Jaccard ≥ 0.8).

| Method & path | Who | |
|---|---|---|
| `GET /api/platform/packs` | any tenant | readable packs `{key, version, root_id, published_at, retired_at}`; the owner sees drafts and retired too |
| `POST /api/platform/packs` | owner | `{key, version, root_id}` → 201 draft. Root of the owner's, no parent (404 / 422 `invalid_target`); taken → 409 `duplicate_name` |
| `POST /api/platform/packs/{key}/{version}/publish` | owner | freeze (idempotent) |
| `POST /api/platform/packs/{key}/{version}/retire` | owner | hide from listing (idempotent; a draft → 422) |
| `GET /api/platform/packs/{key}/{version}/stores` | any tenant | the pack root's subtree `{pack, root_id, stores: [{id, parent_id, kind, name, path, embed_model, chunk_count}]}` ordered by `path` (names from the pack root), so its leaves can go in `leaf_ids`. Same rule as a query: published (retired included) and not excluded; the owner sees drafts; else 404 |
| `GET /api/platform/exclusions` | tenant | the caller's exclusions |
| `PUT /api/platform/exclusions` | tenant | `{key_prefix, reason}` upsert |
| `DELETE /api/platform/exclusions?key_prefix=` | tenant | 204, or 404 when absent |

`{key}` is URL-encoded (`industry%2Fretail`). Owner-only routes answer other tenants `403 forbidden`.

### Migrations

`migrations/` is applied at startup; never edit a shipped file. A migration that writes `stores`
rows and then runs `ALTER TABLE stores` must `SET CONSTRAINTS stores_bump_generation IMMEDIATE`
between the two (or split them), or it fails with 55006 on any non-empty database: see
[`migrations/README.md`](migrations/README.md).

0022 (platform knowledge) adds `platform_packs`, `platform_exclusions` and
`documents.topic_keys` / `supersedes` (`text[]`, default `{}`, GIN-indexed); it writes no rows.

### Search index

Chunk embeddings carry an HNSW index (`vector_cosine_ops`, migration 0009). The dense search runs in
a short transaction with `SET LOCAL hnsw.iterative_scan = relaxed_order` and `hnsw.ef_search =
STRATA_HNSW_EF_SEARCH`: the index returns neighbours before the tenant / store / document filters,
and the iterative scan keeps walking the graph until the filters have passed enough rows, so a leaf
holding 3% of the chunks still gets its full 50 candidates. The rows are re-sorted by distance
afterwards.

That alone fails small leaf sets on a big shared index: the planner estimates the leaf filter at
~5,040 rows whatever its real size, so it always takes the index, and the filtered iterative scan
stops on its memory budget first. At ~1M chunks a 61-chunk tenant leaf set got recall@10 0.66–0.71
and 75–82% short results (`scripts/scale_harness/RESULTS.md`). So each embedding model's leaf set is
first counted (sum of `stores.content_chunks`, kept exact by migration 0020's triggers); at most
`STRATA_EXACT_SCAN_MAX_CHUNKS` (default 5000) runs the same statement with `SET LOCAL
enable_indexscan = off`, i.e. exactly (bitmap on `chunks_store_idx` + sort; recall 1.0, p95 ~6 ms in
the harness), and larger sets keep the HNSW path. `timings_by_model[].path` says which path each
model took (`"exact"` / `"hnsw"`), and `strata_dense_search_groups_total{path}` counts them.

### Re-embedding after a provider switch

A store records the model that made its vectors, and query / ingest refuse a tree whose model differs
from the tenant's current embed provider (`409 embed_model_mismatch`). To move a tree to the new
provider:

```bash
strata reembed --tenant <uuid> --store <root name or uuid> [--batch 256]
```

It embeds every chunk (bulk lane) and every store description with the current provider first, then
in one transaction writes the vectors, recomputes the content / profile rollups and tags the whole
tree with the new model. Queries keep answering 409 until it commits, then 200, in every serving
process too: each process's routing cache checks the tenant's `store_generations` row (bumped at
commit of every store write, migration 0010) on every query and reloads when it moved, so no
restart is needed after a CLI run or a write on another replica. Only the root is
accepted: re-embedding a subtree under an ancestor in the old model would mix vector spaces in the
ancestor's rollup. Memory: about 4 KiB per chunk is held until the transaction.

### Query-embedding cache

Query vectors are kept in an LRU keyed by tenant, embed provider lane, embed model and the
normalised query text (NFC, whitespace collapsed), `STRATA_QUERY_EMBED_CACHE` entries. A repeated
question makes no embed call (and so emits no cost event); `providers.embed.cached` is `true` in its
response. Re-registering a provider starts a fresh lane, so its old vectors are never served.

### Metering

With `METERING_REDIS_URL` set (the one switch), every provider call that does work records one cost
event for billing's stream (`METERING_STREAM`, `XADD ... MAXLEN ~ 100000`), following billing-service
`docs/EMITTER_GUIDE.md`: embed (query and ingest), rerank and chat (`/api/ask`, query expansion).
Health and registration probes, and query-cache hits, record nothing. The deployment's
`metering-redis-url` secret exists, so a deploy of this version records and delivers from the first
request.

Events go through an outbox, the `metering_outbox` table (migration 0011, the commonpractice model):

- **Every paid call gets a row as it returns**, independent of any business transaction: the event
  goes on a bounded in-memory queue (100 000 events, allocated as it fills; the request never waits
  on the database) and one batch writer INSERTs up to 256 events per statement, at most 200 ms after
  the first. An ingest that rolls back, or a reembed that fails or is killed half-way, has already
  recorded every call it paid for. A transient INSERT failure (Postgres down) keeps the batch and
  retries (100 ms doubling to 5 s); a batch Postgres refuses is retried row by row, and a row refused
  on its own is logged with its payload and dropped (counted), so it never blocks later events.
- **Every process flushes the queue before it exits**: a CLI command when it ends; `strata serve`
  after a graceful shutdown flushes FIRST, then stops the drainer after one last pass (10 s + 10 s,
  inside the pod's 30 s `terminationGracePeriodSeconds`).
- **`strata serve` runs the drainer**: it claims due pending rows in one committed statement
  (`FOR UPDATE SKIP LOCKED` plus a 60 s lease on `next_attempt_at`, so any number of replicas drain
  without taking the same row), XADDs them in one pipeline with no database transaction open, then
  marks each row sent or reschedules it (1 s doubling to 5 min). An unreadable row is rescheduled on
  its own; the rest of the batch goes on. The drainer backs off after a failed pass; failures are
  logged at most once per 30 s with their count. Sent rows are deleted after 7 days. A CLI process
  records but never drains: its rows wait for a serving replica. Rows are written whether or not
  Redis is reachable.

Delivery is at least once (a crash between an XADD and the marking re-sends those rows once the
lease runs out); billing deduplicates on `event_key`, which is made once when the call is recorded
and stored in the row. What can still be lost: events still in the in-memory queue when the process
dies WITHOUT a graceful shutdown (crash, OOM kill, SIGKILL) — about the last 200 ms of calls while
Postgres is up, and every call made while Postgres was unreachable — plus events dropped because the
queue was full or Postgres refused the row. A committed row survives crashes, restarts and Redis
outages.

`/health` carries the backlog when metering is on: `metering.outbox_pending`,
`metering.outbox_oldest_pending_age_s` (an old pending row is undelivered spend),
`metering.queued` (in memory, not yet in the outbox) and `metering.dropped_total` (since start).

The attribution headers are taken as sent: strata does not verify `X-Actor-Id` (nor `X-App-Id`,
`X-License-Id`, `X-Project-Id`) against the caller's token, so they are only as trustworthy as the
caller holding a `STRATA_API_TOKENS` token. Billing adjudicates licence and project ownership on
ingest; the actor is recorded as asserted.

| Field | Value |
|---|---|
| `cost_source` | `strata-rag` (the emitting service) |
| `service_key` | `strata-rag-embed` / `strata-rag-rerank` / `strata-rag-chat` |
| `provider`, `model` | short provider name from the endpoint host (`mlx-api`, `voyageai`, `vllm`) and the model sent to it |
| `funding` | `tenant_byo` (+ `usage_extra.byo = true`) for a tenant-registered provider; `local_gpu` for self-hosted endpoints (internal / private hosts, vLLM); else `platform_external` |
| `tenant_id` | the request's tenant |
| `project_id` | `X-Project-Id` when the caller sends it, else billing's tenant-level project id (the tenant uuid's hex digits reversed) |
| `app_id` | `X-App-Id` when sent: the calling app's licences-registry key (`agent`, `care-bus`), relayed verbatim (one value, printable ASCII, at most 128 chars), else empty |
| `actor_id`, `license_id` | `X-Actor-Id`, `X-License-Id` when sent (uuids), else empty |
| `input_tokens`, `output_tokens`, `quantity_unit` | `tokens` for embed and chat (counted with strata's bge-m3 tokenizer; `usage_extra.tokens_counted_by` says so), `passages` for rerank (`usage_extra.tokens` adds the token count) |
| `event_key`, `latency_ms`, `timestamp` | a uuid v7 made when the call is recorded (kept in the outbox row, so every re-delivery carries it), the provider call's wall time, RFC 3339 UTC |

Platform stores: their ingest and re-embed run under the platform owner's token, so the platform
pays. At query time the question embed for platform leaves uses the platform owner's provider for
that pack's model: the event's `tenant_id` is the calling tenant (attribution) and `funding` is
`local_gpu` / `platform_external`, never `tenant_byo`. Rerank and chat stay the caller's.

Query and ask responses carry `providers` next to `timings_ms`: the provider and model of each stage
(`embed`, `rerank`, and `chat` when expansion or generation ran).

### Pages and citations

strata-ingest's bundles mark each page with a `<!-- page:N -->` line. The chunker records the
first and last page each chunk spans (`page_start` / `page_end`; `null` for text without page
anchors, such as audio transcripts, records or plain text, and for chunks ingested before
migration 0007). Every `/api/query` and `/api/ask` result carries them, together with the chunk's
`ord` in its document and the document's `title` and `source`, so a citation can say
"Survey, p. 2–3".

`/api/ask` numbers **documents, not chunks**: one numbered source per document, with every
retrieved chunk of that document (in `ord` order) under the same number, and the document's merged
page span in the label. The prompt shows the model `[n] (store / path · Title · p. 2–3)`; the
answer cites `[n]`.

The response carries `documents`, the numbering map (`number`, `document_id`, `store_path`,
`layer`, `pack?`, `source`, `title`, `page_start`, `page_end`, `chunk_ids`). `citations` stays a
list of **chunk** ids — all the retrieved chunks of each cited document, in first-mention document
order — so a caller that only reads `citations` keeps working. `results` is unchanged: every
retrieved chunk, in rank order.

A caller may pass `instructions` (at most 16 KiB): the job's persona, prepended to the system
message before strata's fixed citation rules. Omit it and the prompt is exactly the default one.

### Tenant eval questions

`POST /api/eval` with no body, or with only `modes` / `beam` / `min_score` /
`allow_foreign_stores`, is unchanged: it scores the server's question file
(`STRATA_EVAL_QUESTIONS`) and stores nothing. A tenant can also keep its own golden questions
(migration 0012, at most 200 per tenant) and score those:

| Route | Body / answer |
|---|---|
| `GET /api/eval/questions` | `{"items":[Question],"next":null}`, oldest first |
| `POST /api/eval/questions` | `QuestionInput` → 201 `Question` |
| `PATCH /api/eval/questions/{id}` | any `QuestionInput` fields (absent = keep, `null` = clear) → `Question` |
| `DELETE /api/eval/questions/{id}` | 204 |
| `POST /api/eval/questions/import` | CSV (`text/csv`) or `{"questions":[QuestionInput]}` → 201 `{"added","total"}`; all or nothing |
| `POST /api/eval` | `{"source":"tenant"}` (all stored), `{"question_ids":[...]}` or `{"questions":[QuestionInput]}` (inline, not stored), plus `root_id` (alias `store_id`), `modes`, `beam`, `min_score` → 202 `EvalRunSummary` (`status: "running"`) + `Location: /api/eval/runs/{id}` |
| `GET /api/eval/runs?limit=&cursor=` | `{"items":[EvalRunSummary],"next"}`, newest first (limit 20, max 100) |
| `GET /api/eval/runs/{id}` | `EvalRun` with `results` |

`QuestionInput` is `{query, expected_document?, expected_leaf?, expected_document_ids?,
expected_chunk_ids?, expected_text?}`; `Question` adds `id, created_by, created_at, updated_at`
(`created_by` is the request's `X-Actor-Id`). A result is a hit when it matches **any** expectation:
`expected_document` (document title, `source`, or the source's file name), one of
`expected_document_ids` / `expected_chunk_ids`, or `expected_text` contained in the chunk text
(case- and whitespace-insensitive). At least one is required. `expected_leaf` (a leaf id, a name
path `lib/sea`, or a unique leaf name) only feeds `leaf_accuracy`. The CSV header names `query`
and any of the expectation columns (any order, other columns ignored); ids in a cell are
separated by `;`, `|` or spaces.

A tenant run is **asynchronous** (migration 0014): one query per question per mode makes a
200-question hybrid run take minutes, longer than a proxy (Cloudflare: ~100 s) keeps a request
open. `POST /api/eval` checks the request and the questions, stores the run as `running` and
answers 202 at once with its summary and `Location: /api/eval/runs/{id}`; a background task,
independent of the client connection, scores it. Poll `GET /api/eval/runs/{id}` (every ~2 s)
until `status` is no longer `running`:

- `running`: `finished_at`, `took_ms`, `verdict`, `reports` are `null`, `results` is `[]`;
  `progress: {done, total}` counts question × mode outcomes scored (written about once a second).
- `completed`: `verdict`, `reports` and all `results`, stored together when scoring ends;
  `progress.done == progress.total`.
- `failed`: `error` says why (a provider error's message; `internal server error` for an
  internal one, detail in the server log), `finished_at` / `took_ms` set, `verdict` / `reports`
  `null`, `results` `[]`.

One run per tenant at a time: a tenant `POST /api/eval` while one runs is 409 `eval_running`
(enforced by a unique index on running runs, so it holds across replicas), and the built-in eval
answers 409 `conflict` as before.

A run's worker beats while it scores: about once a second it stamps the run's `heartbeat_at`
(migration 0015; not part of the API) together with its progress. A `running` run whose heartbeat
is older than `EVAL_STALE_AFTER` (60 s, `src/db/eval.rs`) lost its worker (its process died) and is
marked `failed` with error `interrupted`: swept by every API process at start and every
`EVAL_SWEEP_EVERY` (30 s), and lazily for the tenant on `GET /api/eval/runs`,
`GET /api/eval/runs/{id}` and `POST /api/eval`, so a dead run can neither read `running` nor block
the tenant's next run for longer than the threshold. A run whose worker still beats is never swept,
so a rolling update's new pod leaves the old pod's run alone. A worker whose run was swept stops at
its next beat, and its final write is conditional on `status = 'running'`: the `failed` row stands.
The built-in path (no body / no tenant source) is unchanged: synchronous 200, nothing stored.

`EvalRun` is `{id, source ("tenant"|"inline"), root_id, modes, beam, min_score, status
("running"|"completed"|"failed"), error, progress {done, total}, verdict, started_by, started_at,
finished_at, took_ms, questions, reports}` (`questions` is a snapshot, so editing or deleting a
question never changes an old run; `reports` are the same `ModeReport` rows as the built-in eval:
`recall_at_1/5/10`, `leaf_accuracy`, laya counts), and with `results`
`[{index, question_id, mode, rank, leaf_hit}]` on `GET /api/eval/runs/{id}`.
`EvalRunSummary` (the list item, and the 202 answer) is `EvalRun` without the `questions`
snapshot and `results`, with `question_count` (how many questions the run evaluated) in its place:
`{id, source, root_id, modes, beam, min_score, status, error, progress, verdict, started_by,
started_at, finished_at, took_ms, question_count, reports}`.

Errors: 409 `question_exists` (same query, case-insensitive), 422 `too_many_questions`, 409
`eval_not_ready` (no questions, or a question names a leaf, document or chunk that is not in the
evaluated tree: each is listed), 404 for a `question_ids` entry or `root_id` that is not the
tenant's (all answered by the `POST` itself, before any run is stored), 409 `eval_running` while
another eval of the tenant runs (the built-in eval's own refusal stays 409 `conflict`).

## Development

```bash
cargo build --locked
cargo test --locked          # integration tests start an ephemeral pgvector Postgres
cargo test --doc --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
```

The integration tests use Docker (testcontainers) unless `TEST_DATABASE_URL` points at a
pgvector-enabled database. CI (`.github/workflows/ci-public.yml`) runs the same checks — `cargo fmt`,
`clippy -D warnings`, nextest plus doctests, `cargo audit` — and the two tree scanners (secret scan
and private-reference scan).

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for this
repository's workflow and the
[organisation-wide rules](https://github.com/nonagenticai/.github/blob/main/CONTRIBUTING.md);
test and lint commands are in **Development** above. Commits carry DCO sign-off.

## Licensing

The code in this repository is Apache-2.0 and needs no licence key to build, test or run.
Licence-token authentication is **optional**: it is off while `STRATA_LICENSE_INTROSPECT_URL`
is unset, and static `STRATA_API_TOKENS` keep working. With the URL set, an unknown bearer is
introspected against the licences service and denied if introspection fails.

The licensing stack is public too — [`licenses`](https://github.com/nonagenticai/licenses) (entitlements, plans and tokens) and [`license-authz`](https://github.com/nonagenticai/license-authz) (the Envoy `ext_authz` enforcement point). Neither is required to run this code.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
