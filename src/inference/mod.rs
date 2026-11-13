//! Inference through providers: strata runs no model itself. Embeddings and rerank scores come from
//! HTTP providers (`crate::clients::{embed, rerank}`) behind the [`EmbedProvider`] / [`RerankProvider`]
//! traits; tests plug in the deterministic fakes (`fake`) through the in-process adapters.
//!
//! [`InferenceHandle`] is the cheap, cloneable handle every query / ingest path uses: one tenant's
//! resolved providers (`crate::providers`). Each resolved provider keeps query-before-ingest
//! priority with its own two concurrency limits: query-path calls (a query's embed, its expansion
//! variants, every rerank) take a query slot, ingest sub-batches a bulk slot, so a long ingest can
//! hold at most `bulk_concurrency` provider calls and never the slots a query needs.

pub mod fake;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use tokio::sync::Semaphore;

use crate::error::{Result, StrataError};
use crate::metering::{
    Attribution, CallKind, Metering, ProviderMeta, QuantityUnit, Usage, extra as meter_extra,
};
use crate::routing::BoxFut;
use crate::search::elapsed_ms;
use crate::vecmath::DIM;

pub use fake::{FakeEmbedder, FakeReranker};

/// The embed request's `input_type` (OpenAI extension; providers that ignore it are fine).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    Query,
    Document,
}

/// Dense text embedder: one L2-normalised, `crate::vecmath::DIM`-long vector per text, in order.
pub trait EmbedProvider: Send + Sync {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        input_type: InputType,
    ) -> BoxFut<'a, Result<Vec<Vec<f32>>>>;
}

/// Cross-encoder: one score per passage, in input order.
pub trait RerankProvider: Send + Sync {
    fn rerank<'a>(&'a self, query: &'a str, passages: &'a [String])
    -> BoxFut<'a, Result<Vec<f32>>>;
}

/// Chat completion (spec §4 `Chat`): the trimmed answer text for `messages`. `routed` (routed
/// lanes only) is who the inference-router bills the call to and who asked
/// (`crate::clients::chat::RouterIdentity`); `None` = none of its headers.
pub trait Chat: Send + Sync {
    fn complete<'a>(
        &'a self,
        messages: &'a [crate::clients::chat::ChatMessage],
        max_tokens: u32,
        routed: Option<&'a crate::clients::chat::RouterIdentity>,
    ) -> BoxFut<'a, Result<String>>;

    /// A cheap liveness check for `/health` (no generation).
    fn health(&self) -> BoxFut<'_, Result<()>>;
}

/// A synchronous in-process embedder: the deterministic fakes and test doubles. Run through
/// [`InferenceHandle::in_process`] on the blocking pool, one call at a time.
pub trait Embedder: Send {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// A synchronous in-process reranker (fakes and test doubles).
pub trait Reranker: Send {
    fn score(&mut self, query: &str, passages: &[String]) -> Result<Vec<f32>>;
}

/// Runs a sync `Embedder` / `Reranker` on tokio's blocking pool behind a mutex.
struct InProcess<T: ?Sized>(Arc<Mutex<Box<T>>>);

async fn blocking<T: ?Sized + Send + 'static, R: Send + 'static>(
    inner: &Arc<Mutex<Box<T>>>,
    what: &'static str,
    f: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
) -> Result<R> {
    let inner = Arc::clone(inner);
    tokio::task::spawn_blocking(move || {
        // a panic in an earlier call poisoned the lock; the model object itself is still usable
        let mut guard = inner.lock().unwrap_or_else(|p| p.into_inner());
        f(&mut guard)
    })
    .await
    .map_err(|e| StrataError::Model(format!("{what} panicked: {e}")))?
}

impl EmbedProvider for InProcess<dyn Embedder> {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        _input_type: InputType,
    ) -> BoxFut<'a, Result<Vec<Vec<f32>>>> {
        let texts = texts.to_vec();
        Box::pin(blocking(&self.0, "embedder", move |e| e.embed(&texts)))
    }
}

impl RerankProvider for InProcess<dyn Reranker> {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [String],
    ) -> BoxFut<'a, Result<Vec<f32>>> {
        let (query, passages) = (query.to_string(), passages.to_vec());
        Box::pin(blocking(&self.0, "reranker", move |r| {
            r.score(&query, &passages)
        }))
    }
}

/// Batch size and the two concurrency limits of an [`InferenceHandle`].
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Most texts per ingest embed request (`STRATA_BULK_EMBED_BATCH`).
    pub embed_batch: usize,
    /// Concurrent query-path calls per provider (a provider's `concurrency`).
    pub query_concurrency: usize,
    /// Concurrent ingest embed requests per embed provider (`STRATA_PROVIDER_BULK_CONCURRENCY`).
    pub bulk_concurrency: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            embed_batch: 32,
            query_concurrency: 8,
            bulk_concurrency: 2,
        }
    }
}

/// The store identity of the default in-process embedder (`stores.embed_model`).
pub const DEFAULT_EMBED_MODEL_ID: &str = "BAAI/bge-m3@1024";

/// One resolved embed provider with its own query and bulk slots, and the identity its vectors
/// are recorded under (`stores.embed_model`).
/// Source of [`EmbedLane::id`]s.
static NEXT_LANE_ID: AtomicU64 = AtomicU64::new(1);

pub struct EmbedLane {
    /// Unique per lane built in this process: a re-registered provider gets a new lane, so state
    /// keyed by it (the query-embedding cache) never outlives the provider that produced it.
    id: u64,
    provider: Arc<dyn EmbedProvider>,
    model_id: String,
    /// What a cost event names as provider / model / funding (`crate::metering`).
    meta: ProviderMeta,
    limits: Limits,
    batch: usize,
    query_slots: Semaphore,
    bulk_slots: Semaphore,
}

impl EmbedLane {
    /// An in-process provider until [`Self::with_meta`] says otherwise.
    pub fn new(
        provider: Arc<dyn EmbedProvider>,
        model_id: impl Into<String>,
        limits: Limits,
    ) -> Self {
        let model_id = model_id.into();
        EmbedLane {
            id: NEXT_LANE_ID.fetch_add(1, Ordering::Relaxed),
            provider,
            meta: ProviderMeta::in_process(&model_id),
            model_id,
            limits,
            batch: limits.embed_batch.max(1),
            query_slots: Semaphore::new(limits.query_concurrency.max(1)),
            bulk_slots: Semaphore::new(limits.bulk_concurrency.max(1)),
        }
    }

    pub fn with_meta(self, meta: ProviderMeta) -> Self {
        EmbedLane { meta, ..self }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn meta(&self) -> &ProviderMeta {
        &self.meta
    }
}

/// One resolved rerank provider with its own query slots.
pub struct RerankLane {
    provider: Arc<dyn RerankProvider>,
    meta: ProviderMeta,
    query_slots: Semaphore,
}

impl RerankLane {
    /// An in-process provider until [`Self::with_meta`] says otherwise.
    pub fn new(provider: Arc<dyn RerankProvider>, query_concurrency: usize) -> Self {
        RerankLane {
            provider,
            meta: ProviderMeta::in_process("in-process-reranker"),
            query_slots: Semaphore::new(query_concurrency.max(1)),
        }
    }

    pub fn with_meta(self, meta: ProviderMeta) -> Self {
        RerankLane { meta, ..self }
    }

    pub fn meta(&self) -> &ProviderMeta {
        &self.meta
    }
}

/// A handle's metering: the emitter and who its calls are billed to.
struct Meter {
    metering: Arc<Metering>,
    attr: Attribution,
    /// The platform pays these calls (`ProviderMeta::platform_funded`), see
    /// [`InferenceHandle::platform_funded`].
    platform_funded: bool,
}

impl Meter {
    /// What the cost event of a call on `lane` names as provider / model / funding.
    fn meta(&self, lane: &ProviderMeta) -> ProviderMeta {
        if self.platform_funded {
            lane.platform_funded()
        } else {
            lane.clone()
        }
    }
}

/// Cheap, cloneable handle to one tenant's resolved embed + rerank providers. A capability with no
/// provider (`None`) fails its calls with `409 provider_not_configured`, so a path that never
/// needs it (a store create needs no reranker) still works.
#[derive(Clone)]
pub struct InferenceHandle {
    embed: Option<Arc<EmbedLane>>,
    rerank: Option<Arc<RerankLane>>,
    chat: Option<Arc<ChatLane>>,
    /// A capability whose registered provider cannot be used (a refused stored host, a key that
    /// does not decrypt): raised only when that capability is used, so one bad registration
    /// never fails a request that does not need it.
    embed_fault: Option<LaneFault>,
    rerank_fault: Option<LaneFault>,
    chat_fault: Option<LaneFault>,
    /// `None`: calls are not metered (metering off, health probes, tests).
    meter: Option<Arc<Meter>>,
}

/// Why a capability's registered provider cannot be used for this request (cloneable: the error
/// is rebuilt each time the lane is used).
#[derive(Clone, Debug)]
pub enum LaneFault {
    /// 422 with this code (e.g. `provider_host_not_allowed`).
    Rejected { code: &'static str, message: String },
    /// 500 `internal` (e.g. a key that does not decrypt); the detail is only logged.
    Internal(String),
}

impl LaneFault {
    pub fn of(e: &StrataError) -> Self {
        match e {
            StrataError::ProviderRejected { code, message } => LaneFault::Rejected {
                code,
                message: message.clone(),
            },
            other => LaneFault::Internal(other.to_string()),
        }
    }

    fn error(&self) -> StrataError {
        match self {
            LaneFault::Rejected { code, message } => StrataError::ProviderRejected {
                code,
                message: message.clone(),
            },
            LaneFault::Internal(m) => StrataError::Config(m.clone()),
        }
    }
}

/// Longest wait for a chat slot (one of the lane's concurrent slots). A slot holder may be running
/// a full answer (up to 8192 tokens) at about 7.5 tok/s on the slow local model, i.e. minutes, so the
/// wait must outlast the deployment chat timeout (240 s) rather than track the expansion bound.
/// Deliberately its own constant, not `crate::expand::EXPAND_TIMEOUT`.
pub const CHAT_SLOT_WAIT: std::time::Duration = std::time::Duration::from_secs(300);

/// One resolved chat provider with its own slots (`concurrency`).
pub struct ChatLane {
    provider: Arc<dyn Chat>,
    meta: ProviderMeta,
    slots: Semaphore,
    /// The provider is the inference-router (`[default.chat] routed = true`, deployment default
    /// only): it bills the call to the tenant strata names, so strata emits no chat event.
    routed: bool,
}

impl ChatLane {
    /// An in-process provider until [`Self::with_meta`] says otherwise.
    pub fn new(provider: Arc<dyn Chat>, concurrency: usize) -> Self {
        ChatLane {
            provider,
            meta: ProviderMeta::in_process("in-process-chat"),
            slots: Semaphore::new(concurrency.max(1)),
            routed: false,
        }
    }

    pub fn with_meta(self, meta: ProviderMeta) -> Self {
        ChatLane { meta, ..self }
    }

    /// Sets the lane's `routed` flag; only the deployment's default chat lane is ever routed.
    pub fn with_routed(self, routed: bool) -> Self {
        ChatLane { routed, ..self }
    }

    pub fn is_routed(&self) -> bool {
        self.routed
    }

    pub fn meta(&self) -> &ProviderMeta {
        &self.meta
    }
}

impl std::fmt::Debug for InferenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceHandle")
            .field("embed", &self.embed.as_ref().map(|l| l.model_id.as_str()))
            .field("rerank", &self.rerank.is_some())
            .field("chat", &self.chat.is_some())
            .field("embed_fault", &self.embed_fault)
            .field("rerank_fault", &self.rerank_fault)
            .field("chat_fault", &self.chat_fault)
            .field("metered", &self.meter.is_some())
            .finish()
    }
}

fn check_vectors(vectors: Vec<Vec<f32>>, expected: usize) -> Result<Vec<Vec<f32>>> {
    if vectors.len() != expected {
        return Err(StrataError::Model(format!(
            "embedder returned {} vectors for {expected} inputs",
            vectors.len()
        )));
    }
    if let Some(v) = vectors.iter().find(|v| v.len() != DIM) {
        return Err(StrataError::Model(format!(
            "embedder returned a {}-dim vector, expected {DIM}",
            v.len()
        )));
    }
    Ok(vectors)
}

impl InferenceHandle {
    /// Both providers, each with its own slots sized by `limits`; the embedder's identity is
    /// [`DEFAULT_EMBED_MODEL_ID`] (see [`Self::with_embed_model_id`]).
    pub fn new(
        embedder: Arc<dyn EmbedProvider>,
        reranker: Arc<dyn RerankProvider>,
        limits: Limits,
    ) -> Self {
        Self {
            embed: Some(Arc::new(EmbedLane::new(
                embedder,
                DEFAULT_EMBED_MODEL_ID,
                limits,
            ))),
            chat: None,
            rerank: Some(Arc::new(RerankLane::new(
                reranker,
                limits.query_concurrency,
            ))),
            embed_fault: None,
            rerank_fault: None,
            chat_fault: None,
            meter: None,
        }
    }

    /// Resolved lanes; `None` = that capability is not configured.
    pub fn from_lanes(
        embed: Option<Arc<EmbedLane>>,
        rerank: Option<Arc<RerankLane>>,
        chat: Option<Arc<ChatLane>>,
    ) -> Self {
        Self {
            embed,
            rerank,
            chat,
            embed_fault: None,
            rerank_fault: None,
            chat_fault: None,
            meter: None,
        }
    }

    /// The same handle with `capability` ("embed" | "rerank" | "chat") unusable for `fault`.
    pub fn with_fault(self, capability: &str, fault: LaneFault) -> Self {
        match capability {
            "embed" => Self {
                embed: None,
                embed_fault: Some(fault),
                ..self
            },
            "rerank" => Self {
                rerank: None,
                rerank_fault: Some(fault),
                ..self
            },
            _ => Self {
                chat: None,
                chat_fault: Some(fault),
                ..self
            },
        }
    }

    /// The same embed / rerank providers plus `chat` (default concurrency).
    pub fn with_chat(self, chat: Arc<dyn Chat>) -> Self {
        Self {
            chat: Some(Arc::new(ChatLane::new(
                chat,
                Limits::default().query_concurrency,
            ))),
            ..self
        }
    }
    /// The same providers, with every embed / rerank / chat call metered to `attr` (probes excepted).
    pub fn metered(self, metering: Arc<Metering>, attr: Attribution) -> Self {
        Self {
            meter: Some(Arc::new(Meter {
                metering,
                attr,
                platform_funded: false,
            })),
            ..self
        }
    }

    /// The same metered providers with every call's funding the platform's (spec
    /// platform-knowledge §7: the question's embed for platform leaves runs on the platform
    /// owner's provider, attributed to the calling tenant, never `tenant_byo`). Unmetered: as is.
    pub fn platform_funded(self) -> Self {
        let meter = self.meter.as_deref().map(|m| {
            Arc::new(Meter {
                metering: Arc::clone(&m.metering),
                attr: m.attr.clone(),
                platform_funded: true,
            })
        });
        Self { meter, ..self }
    }

    pub fn is_metered(&self) -> bool {
        self.meter.is_some()
    }

    /// Provider / model / funding of the embed lane (`None` = not configured).
    pub fn embed_provider(&self) -> Option<&ProviderMeta> {
        self.embed.as_deref().map(EmbedLane::meta)
    }

    /// Provider / model / funding of the rerank lane (`None` = not configured).
    pub fn rerank_provider(&self) -> Option<&ProviderMeta> {
        self.rerank.as_deref().map(RerankLane::meta)
    }

    /// Provider / model / funding of the chat lane (`None` = not configured).
    pub fn chat_provider(&self) -> Option<&ProviderMeta> {
        self.chat.as_deref().map(ChatLane::meta)
    }

    /// The same providers with the embedder recorded under `id` (tests simulating a model change).
    pub fn with_embed_model_id(self, id: impl Into<String>) -> Self {
        let embed = self.embed.map(|l| {
            Arc::new(EmbedLane::new(l.provider.clone(), id, l.limits).with_meta(l.meta.clone()))
        });
        Self { embed, ..self }
    }

    /// In-process sync models (fakes, test doubles), default limits.
    pub fn in_process(embedder: Box<dyn Embedder>, reranker: Box<dyn Reranker>) -> Self {
        Self::new(
            Arc::new(InProcess(Arc::new(Mutex::new(embedder)))),
            Arc::new(InProcess(Arc::new(Mutex::new(reranker)))),
            Limits::default(),
        )
    }

    /// `FakeEmbedder` + `FakeReranker` in-process, for tests: no network.
    pub fn fake() -> Self {
        Self::in_process(Box::new(FakeEmbedder), Box::new(FakeReranker))
    }

    pub fn has_embed(&self) -> bool {
        self.embed.is_some()
    }

    pub fn has_rerank(&self) -> bool {
        self.rerank.is_some()
    }

    pub fn has_chat(&self) -> bool {
        self.chat.is_some()
    }

    /// `Ok` when chat can be used: else its fault, or 409 `provider_not_configured`.
    pub fn require_chat(&self) -> Result<()> {
        self.chat_lane().map(|_| ())
    }

    fn chat_lane(&self) -> Result<&ChatLane> {
        if let Some(f) = &self.chat_fault {
            return Err(f.error());
        }
        self.chat
            .as_deref()
            .ok_or(StrataError::ProviderNotConfigured { capability: "chat" })
    }

    /// One completion. A provider failure is `GenerationUnavailable` (503) carrying the provider
    /// error class (`auth`, `retryable`, ...); no provider is 409 `provider_not_configured`.
    pub async fn chat(
        &self,
        messages: &[crate::clients::chat::ChatMessage],
        max_tokens: u32,
    ) -> Result<String> {
        let lane = self.chat_lane()?;
        let _slot = match tokio::time::timeout(CHAT_SLOT_WAIT, lane.slots.acquire()).await {
            Ok(slot) => {
                slot.map_err(|_| StrataError::Model("inference handle closed".to_string()))?
            }
            Err(_) => {
                return Err(StrataError::GenerationUnavailable(
                    "chat: retryable: every chat slot is busy".to_string(),
                ));
            }
        };
        // a routed lane names the billed tenant, project and the caller's app / actor to the
        // router; an unmetered call (no tenant to bill) names none
        let routed = if lane.routed {
            self.meter
                .as_deref()
                .map(|m| crate::clients::chat::RouterIdentity {
                    tenant: m.attr.tenant.id(),
                    project: m.attr.project_id(),
                    app: m.attr.caller.app_id.clone(),
                    actor: m.attr.caller.actor_id,
                })
        } else {
            None
        };
        let t = Instant::now();
        let reply = lane
            .provider
            .complete(messages, max_tokens, routed.as_ref())
            .await
            .map_err(|e| match e {
                StrataError::Upstream { message, .. } => {
                    StrataError::GenerationUnavailable(format!("chat: {message}"))
                }
                other => other,
            })?;
        // routed: the inference-router meters the call from vLLM's real usage; a strata event too
        // would bill it twice
        if !lane.routed
            && let Some(m) = self.meter.as_deref()
        {
            // the chat client returns no provider usage: tokens counted here
            m.metering.emit(
                &m.attr,
                &m.meta(&lane.meta),
                Usage {
                    call: CallKind::Chat,
                    input: m
                        .metering
                        .tokens(messages.iter().map(|msg| msg.content.as_str())),
                    output: m.metering.tokens([reply.as_str()]),
                    unit: QuantityUnit::Tokens,
                    latency_ms: elapsed_ms(t),
                    extra: meter_extra(&[("max_tokens", json!(max_tokens))], true),
                },
            );
        }
        Ok(reply)
    }

    /// `/health`: the chat provider's liveness check, slot-free like `probe_embed`.
    pub async fn probe_chat(&self) -> Result<()> {
        self.chat_lane()?.provider.health().await
    }

    fn embed_lane(&self) -> Result<&EmbedLane> {
        if let Some(f) = &self.embed_fault {
            return Err(f.error());
        }
        self.embed
            .as_deref()
            .ok_or(StrataError::ProviderNotConfigured {
                capability: "embed",
            })
    }

    fn rerank_lane(&self) -> Result<&RerankLane> {
        if let Some(f) = &self.rerank_fault {
            return Err(f.error());
        }
        self.rerank
            .as_deref()
            .ok_or(StrataError::ProviderNotConfigured {
                capability: "rerank",
            })
    }

    /// The identity this embedder's vectors are recorded under (`stores.embed_model`).
    pub fn embed_model_id(&self) -> Result<&str> {
        Ok(self.embed_lane()?.model_id())
    }

    /// The embed lane's process-unique id (see `EmbedLane::id`).
    pub fn embed_lane_id(&self) -> Result<u64> {
        Ok(self.embed_lane()?.id())
    }

    async fn embed_in(
        lane: &EmbedLane,
        texts: &[String],
        input_type: InputType,
        slots: &Semaphore,
        meter: Option<&Meter>,
    ) -> Result<Vec<Vec<f32>>> {
        let _slot = slots
            .acquire()
            .await
            .map_err(|_| StrataError::Model("inference handle closed".to_string()))?;
        let t = Instant::now();
        let vectors = check_vectors(lane.provider.embed(texts, input_type).await?, texts.len())?;
        if let Some(m) = meter {
            m.metering.emit(
                &m.attr,
                &m.meta(&lane.meta),
                Usage {
                    call: CallKind::Embed,
                    input: m.metering.tokens(texts.iter().map(String::as_str)),
                    output: 0,
                    unit: QuantityUnit::Tokens,
                    latency_ms: elapsed_ms(t),
                    extra: meter_extra(
                        &[
                            ("texts", json!(texts.len())),
                            ("input_type", json!(input_type)),
                        ],
                        true,
                    ),
                },
            );
        }
        Ok(vectors)
    }

    /// `texts` split into batch-sized requests, all issued at once, each waiting for a slot of
    /// the lane's query or bulk semaphore; vectors come back in input order. The first failure
    /// fails the whole call.
    async fn embed_batched(
        &self,
        texts: Vec<String>,
        input_type: InputType,
        bulk: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let lane = self.embed_lane()?;
        let slots = if bulk {
            &lane.bulk_slots
        } else {
            &lane.query_slots
        };
        let meter = self.meter.as_deref();
        let parts = futures_util::future::try_join_all(
            texts
                .chunks(lane.batch)
                .map(|part| Self::embed_in(lane, part, input_type, slots, meter)),
        )
        .await?;
        Ok(parts.into_iter().flatten().collect())
    }

    /// Document vectors for ingest (bulk lane, `input_type: document`), in input order.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_batched(texts, InputType::Document, true).await
    }

    /// Query-path vectors for several texts (query expansion variants), in input order.
    pub async fn embed_queries(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_batched(texts, InputType::Query, false).await
    }

    /// One query-path vector (`input_type: query`).
    pub async fn embed_one(&self, text: String) -> Result<Vec<f32>> {
        let lane = self.embed_lane()?;
        let mut vectors = Self::embed_in(
            lane,
            &[text],
            InputType::Query,
            &lane.query_slots,
            self.meter.as_deref(),
        )
        .await?;
        vectors
            .pop()
            .ok_or_else(|| StrataError::Model("embedder returned no vector".to_string()))
    }

    /// One embed for `/health`, straight to the provider: takes no query or bulk slot, so a probe
    /// can neither wait behind nor take capacity from real traffic (the engine bounds probes itself).
    pub async fn probe_embed(&self) -> Result<Vec<f32>> {
        let texts = ["health check".to_string()];
        let lane = self.embed_lane()?;
        let mut v = check_vectors(lane.provider.embed(&texts, InputType::Query).await?, 1)?;
        v.pop()
            .ok_or_else(|| StrataError::Model("embedder returned no vector".to_string()))
    }

    /// One single-passage rerank for `/health`, slot-free like `probe_embed`.
    pub async fn probe_rerank(&self) -> Result<Vec<f32>> {
        let passages = ["health check".to_string()];
        let scores = self
            .rerank_lane()?
            .provider
            .rerank("health check", &passages)
            .await?;
        if scores.len() != 1 {
            return Err(StrataError::Model(format!(
                "reranker returned {} scores for 1 passage",
                scores.len()
            )));
        }
        Ok(scores)
    }

    /// Scores for `passages` in input order; query lane.
    pub async fn rerank(&self, query: String, passages: Vec<String>) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let lane = self.rerank_lane()?;
        let _slot = lane
            .query_slots
            .acquire()
            .await
            .map_err(|_| StrataError::Model("inference handle closed".to_string()))?;
        let t = Instant::now();
        let scores = lane.provider.rerank(&query, &passages).await?;
        if scores.len() != passages.len() {
            return Err(StrataError::Model(format!(
                "reranker returned {} scores for {} passages",
                scores.len(),
                passages.len()
            )));
        }
        if let Some(m) = self.meter.as_deref() {
            let tokens = m
                .metering
                .tokens(std::iter::once(query.as_str()).chain(passages.iter().map(String::as_str)));
            m.metering.emit(
                &m.attr,
                &m.meta(&lane.meta),
                Usage {
                    call: CallKind::Rerank,
                    input: passages.len() as u64,
                    output: 0,
                    unit: QuantityUnit::Passages,
                    latency_ms: elapsed_ms(t),
                    extra: meter_extra(&[("tokens", json!(tokens))], true),
                },
            );
        }
        Ok(scores)
    }
}

#[cfg(test)]
pub(crate) mod tests;
