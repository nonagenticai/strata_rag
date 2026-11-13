use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::error::StrataError;
use crate::vecmath::DIM;

fn texts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("text number {i}")).collect()
}

/// Async test provider: records every call, holds `Document` calls on `gate` while `gated`, and
/// tracks how many calls are in flight at once.
struct Probe {
    calls: Mutex<Vec<(InputType, usize)>>,
    gate: tokio::sync::Semaphore,
    gated: std::sync::atomic::AtomicBool,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl Default for Probe {
    fn default() -> Self {
        Probe {
            calls: Mutex::default(),
            gate: tokio::sync::Semaphore::new(0),
            gated: Default::default(),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        }
    }
}

impl Probe {
    fn gated() -> Arc<Self> {
        let p = Probe::default();
        p.gated.store(true, Ordering::SeqCst);
        Arc::new(p)
    }
    fn calls(&self) -> Vec<(InputType, usize)> {
        self.calls.lock().unwrap().clone()
    }
    fn open(&self) {
        self.gated.store(false, Ordering::SeqCst);
        self.gate.add_permits(1 << 20);
    }
}

impl EmbedProvider for Probe {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        input_type: InputType,
    ) -> BoxFut<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push((input_type, texts.len()));
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            if input_type == InputType::Document && self.gated.load(Ordering::SeqCst) {
                self.gate.acquire().await.unwrap().forget();
            }
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            FakeEmbedder.embed(texts)
        })
    }
}

impl RerankProvider for Probe {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [String],
    ) -> BoxFut<'a, Result<Vec<f32>>> {
        Box::pin(async move { FakeReranker.score(query, passages) })
    }
}

fn handle(p: &Arc<Probe>, embed_batch: usize, bulk: usize) -> InferenceHandle {
    InferenceHandle::new(
        p.clone(),
        p.clone(),
        Limits {
            embed_batch,
            query_concurrency: 4,
            bulk_concurrency: bulk,
        },
    )
}

/// Wait until `p` has seen `n` calls (bounded so a broken test fails instead of hanging).
async fn until_calls(p: &Probe, n: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while p.calls().len() < n {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("provider calls arrived");
}

#[tokio::test]
async fn an_oversized_ingest_is_split_into_batches_and_keeps_order() {
    let p = Arc::new(Probe::default());
    let h = handle(&p, 4, 2);
    let all = texts(10);
    let got = h.embed(all.clone()).await.unwrap();
    let mut sizes: Vec<usize> = p.calls().iter().map(|c| c.1).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![2, 4, 4]);
    assert!(p.calls().iter().all(|c| c.0 == InputType::Document));
    assert_eq!(got, FakeEmbedder.embed(&all).unwrap());
}

#[tokio::test]
async fn bulk_calls_never_exceed_the_bulk_limit() {
    let p = Probe::gated();
    let h = handle(&p, 1, 2);
    let ingest = tokio::spawn({
        let h = h.clone();
        async move { h.embed(texts(6)).await }
    });
    until_calls(&p, 2).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        p.calls().len(),
        2,
        "a third bulk call started past the limit"
    );
    p.open();
    assert_eq!(ingest.await.unwrap().unwrap().len(), 6);
    assert_eq!(p.max_in_flight.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_query_embed_and_rerank_run_while_ingest_holds_every_bulk_slot() {
    let p = Probe::gated();
    let h = handle(&p, 1, 2);
    let ingest = tokio::spawn({
        let h = h.clone();
        async move { h.embed(texts(8)).await }
    });
    until_calls(&p, 2).await;
    let q = tokio::time::timeout(Duration::from_secs(2), async {
        let v = h.embed_one("a query".to_string()).await.unwrap();
        let s = h
            .rerank("a query".to_string(), vec!["a query here".to_string()])
            .await
            .unwrap();
        (v, s)
    })
    .await
    .expect("the query path must not wait for ingest");
    assert_eq!(
        q.0,
        FakeEmbedder.embed(&["a query".to_string()]).unwrap()[0]
    );
    assert_eq!(q.1, vec![1.0]);
    assert!(p.calls().contains(&(InputType::Query, 1)));
    p.open();
    assert_eq!(ingest.await.unwrap().unwrap().len(), 8);
}

#[tokio::test]
async fn embed_queries_uses_the_query_input_type() {
    let p = Arc::new(Probe::default());
    let h = handle(&p, 32, 1);
    let got = h.embed_queries(texts(3)).await.unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(p.calls(), vec![(InputType::Query, 3)]);
}

struct Short;

impl EmbedProvider for Short {
    fn embed<'a>(&'a self, _: &'a [String], _: InputType) -> BoxFut<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async { Ok(vec![vec![0.0; DIM]]) })
    }
}

impl RerankProvider for Short {
    fn rerank<'a>(&'a self, _: &'a str, _: &'a [String]) -> BoxFut<'a, Result<Vec<f32>>> {
        Box::pin(async { Ok(vec![0.5]) })
    }
}

#[tokio::test]
async fn a_provider_returning_the_wrong_count_is_an_error() {
    let s = Arc::new(Short);
    let h = InferenceHandle::new(s.clone(), s, Limits::default());
    match h.embed(texts(2)).await {
        Err(StrataError::Model(m)) => assert!(m.contains("1 vectors for 2 inputs"), "{m}"),
        other => panic!("expected a count error, got {other:?}"),
    }
    match h.rerank("q".into(), texts(3)).await {
        Err(StrataError::Model(m)) => assert!(m.contains("1 scores for 3 passages"), "{m}"),
        other => panic!("expected a count error, got {other:?}"),
    }
}

struct Failing(Arc<AtomicUsize>);

impl Embedder for Failing {
    fn embed(&mut self, _: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(StrataError::Upstream {
            service: "embed",
            message: "retryable: status 503".into(),
        })
    }
}

#[tokio::test]
async fn a_provider_error_reaches_the_caller_unchanged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let h = InferenceHandle::in_process(Box::new(Failing(calls.clone())), Box::new(FakeReranker));
    let err = h.embed(texts(3)).await.unwrap_err();
    assert!(
        matches!(&err, StrataError::Upstream { service: "embed", message } if message == "retryable: status 503"),
        "{err:?}"
    );
}

/// Panics on the text "panic"; behaves like FakeEmbedder otherwise.
struct Panicky;

impl Embedder for Panicky {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.iter().any(|t| t == "panic") {
            panic!("embedder exploded");
        }
        FakeEmbedder.embed(texts)
    }
}

#[tokio::test]
async fn a_panicking_in_process_embedder_errors_and_the_handle_survives() {
    let h = InferenceHandle::in_process(Box::new(Panicky), Box::new(FakeReranker));
    match h.embed_one("panic".to_string()).await {
        Err(StrataError::Model(m)) => assert!(m.contains("panicked"), "{m}"),
        other => panic!("expected a panic error, got {other:?}"),
    }
    assert_eq!(h.embed_one("fine".to_string()).await.unwrap().len(), DIM);
}

#[tokio::test]
async fn empty_inputs_short_circuit_and_the_fake_handle_matches_the_fakes() {
    let fake = InferenceHandle::fake();
    assert!(fake.embed(Vec::new()).await.unwrap().is_empty());
    assert!(fake.embed_queries(Vec::new()).await.unwrap().is_empty());
    assert!(
        fake.rerank("q".into(), Vec::new())
            .await
            .unwrap()
            .is_empty()
    );
    let t = texts(40);
    assert_eq!(
        fake.embed(t.clone()).await.unwrap(),
        FakeEmbedder.embed(&t).unwrap()
    );
    let one = fake.embed_one("alpha beta".into()).await.unwrap();
    assert_eq!(
        one,
        FakeEmbedder.embed(&["alpha beta".to_string()]).unwrap()[0]
    );
    let passages = vec!["alpha".to_string(), "alpha beta".to_string()];
    assert_eq!(
        fake.rerank("Alpha beta".into(), passages.clone())
            .await
            .unwrap(),
        FakeReranker.score("Alpha beta", &passages).unwrap()
    );
}

/// Blocks every call whose first text is "hold" until `release` is notified.
struct Holding {
    entered: AtomicUsize,
    release: tokio::sync::Notify,
}

impl EmbedProvider for Holding {
    fn embed<'a>(&'a self, texts: &'a [String], _: InputType) -> BoxFut<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            if texts.first().is_some_and(|t| t == "hold") {
                self.entered.fetch_add(1, Ordering::SeqCst);
                self.release.notified().await;
            }
            FakeEmbedder.embed(texts)
        })
    }
}

impl RerankProvider for Holding {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [String],
    ) -> BoxFut<'a, Result<Vec<f32>>> {
        Box::pin(async move { FakeReranker.score(query, passages) })
    }
}

#[tokio::test]
async fn health_probes_take_no_query_slot() {
    let p = Arc::new(Holding {
        entered: AtomicUsize::new(0),
        release: tokio::sync::Notify::new(),
    });
    let h = InferenceHandle::new(
        p.clone(),
        p.clone(),
        Limits {
            embed_batch: 8,
            query_concurrency: 1,
            bulk_concurrency: 1,
        },
    );
    let held = tokio::spawn({
        let h = h.clone();
        async move { h.embed_one("hold".to_string()).await }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while p.entered.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the query took the only slot");
    // the only query slot is taken: a query waits ...
    assert!(
        tokio::time::timeout(Duration::from_millis(100), h.embed_one("x".to_string()))
            .await
            .is_err(),
        "the query slot was not actually held"
    );
    // ... the probes do not
    let probes = tokio::time::timeout(Duration::from_secs(1), async {
        (h.probe_embed().await, h.probe_rerank().await)
    })
    .await
    .expect("probes must not wait for a query slot");
    assert_eq!(probes.0.unwrap().len(), DIM);
    assert_eq!(probes.1.unwrap().len(), 1);
    p.release.notify_waiters();
    held.await.unwrap().unwrap();
}

/// The router identity headers one chat request carried (`None` = header absent).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SeenIdentity {
    pub tenant: Option<String>,
    pub project: Option<String>,
    pub app: Option<String>,
    pub actor: Option<String>,
}

/// A chat server recording each request's router identity headers.
pub(crate) async fn tenant_header_mock() -> (String, Arc<Mutex<Vec<SeenIdentity>>>) {
    use crate::clients::chat::{
        ACTOR_ID_HEADER, APP_ID_HEADER, EFFECTIVE_TENANT_HEADER, PROJECT_ID_HEADER,
    };
    use axum::http::HeaderMap;
    use axum::routing::post;
    let seen: Arc<Mutex<Vec<SeenIdentity>>> = Arc::default();
    let s = seen.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(move |h: HeaderMap| {
            let s = s.clone();
            async move {
                let get = |name: &str| h.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
                s.lock().unwrap().push(SeenIdentity {
                    tenant: get(EFFECTIVE_TENANT_HEADER),
                    project: get(PROJECT_ID_HEADER),
                    app: get(APP_ID_HEADER),
                    actor: get(ACTOR_ID_HEADER),
                });
                axum::Json(json!({"choices": [{"message": {"content": "an answer"}}]}))
            }
        }),
    );
    (
        format!("{}/v1", crate::clients::testutil::spawn(app).await),
        seen,
    )
}

const ROUTED_TENANT: &str = "0195f48b-d000-7000-8000-000000000177";
const CALLER_PROJECT: &str = "0195f48b-d000-7000-8000-0000000001a1";
const CALLER_ACTOR: &str = "0195f48b-d000-7000-8000-0000000001a2";

/// The attribution an agent request to `/api/ask` carries.
fn agent_caller() -> crate::metering::Caller {
    crate::metering::Caller {
        project_id: Some(CALLER_PROJECT.parse().unwrap()),
        app_id: Some("agent".to_string()),
        actor_id: Some(CALLER_ACTOR.parse().unwrap()),
        license_id: None,
    }
}

/// A chat lane at `base` (deployment-default funding), `routed` as given, metered to
/// [`ROUTED_TENANT`] with `caller` into a capture sink when `caller` is given.
fn chat_handle(
    base: &str,
    routed: bool,
    caller: Option<crate::metering::Caller>,
) -> (
    InferenceHandle,
    tokio::sync::mpsc::Receiver<crate::metering::CostEvent>,
) {
    let client = crate::clients::chat::ChatClient::new(
        base,
        None,
        "local-llm",
        2000,
        crate::clients::chat::ChatOptions::default(),
    )
    .unwrap();
    let lane = ChatLane::new(Arc::new(client), 2)
        .with_meta(ProviderMeta::deployment_default(base, "local-llm"))
        .with_routed(routed);
    let h = InferenceHandle::from_lanes(None, None, Some(Arc::new(lane)));
    let (m, rx) = crate::metering::Metering::capture(Arc::new(crate::text::WhitespaceCounter));
    let h = match caller {
        Some(caller) => h.metered(
            m,
            Attribution {
                tenant: crate::tenant::Tenant::parse(ROUTED_TENANT).unwrap(),
                caller,
            },
        ),
        None => h,
    };
    (h, rx)
}

fn ask() -> Vec<crate::clients::chat::ChatMessage> {
    vec![crate::clients::chat::ChatMessage::new(
        "user",
        "what is light?",
    )]
}

/// Routed (the inference-router): the billed tenant goes out as `X-Effective-Tenant-Id`, the
/// caller's project / app / actor are relayed, and strata emits no `strata-rag-chat` event, the
/// router bills the call.
#[tokio::test]
async fn a_routed_chat_names_the_tenant_and_relays_the_callers_identity() {
    let (base, seen) = tenant_header_mock().await;
    let (h, mut rx) = chat_handle(&base, true, Some(agent_caller()));
    assert_eq!(h.chat(&ask(), 16).await.unwrap(), "an answer");
    assert_eq!(
        seen.lock().unwrap().clone(),
        vec![SeenIdentity {
            tenant: Some(ROUTED_TENANT.to_string()),
            project: Some(CALLER_PROJECT.to_string()),
            app: Some("agent".to_string()),
            actor: Some(CALLER_ACTOR.to_string()),
        }]
    );
    assert!(
        matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ),
        "a routed chat call must not emit a strata-rag-chat event"
    );
}

/// A caller that named no project, app or actor: the project is the tenant-level one (billing's
/// rule for project-less callers), and app / actor stay ABSENT, never strata's own identity.
#[tokio::test]
async fn a_routed_chat_for_a_bare_caller_sends_the_tenant_level_project_and_no_app_or_actor() {
    let (base, seen) = tenant_header_mock().await;
    let (h, _rx) = chat_handle(&base, true, Some(crate::metering::Caller::default()));
    h.chat(&ask(), 16).await.unwrap();
    let tenant: uuid::Uuid = ROUTED_TENANT.parse().unwrap();
    assert_eq!(
        seen.lock().unwrap().clone(),
        vec![SeenIdentity {
            tenant: Some(ROUTED_TENANT.to_string()),
            project: Some(crate::metering::tenant_level_project_id(tenant).to_string()),
            app: None,
            actor: None,
        }]
    );
}

/// Routed but unmetered (no tenant to bill): no identity header at all.
#[tokio::test]
async fn an_unmetered_routed_chat_sends_no_tenant_header() {
    let (base, seen) = tenant_header_mock().await;
    let (h, _rx) = chat_handle(&base, true, None);
    h.chat(&ask(), 16).await.unwrap();
    assert_eq!(seen.lock().unwrap().clone(), vec![SeenIdentity::default()]);
}

/// Not routed (regression guard): none of the router's identity headers reach the provider, even
/// with a fully attributed caller, and strata emits the chat event.
#[tokio::test]
async fn a_direct_chat_sends_no_tenant_header_and_emits_the_chat_event() {
    let (base, seen) = tenant_header_mock().await;
    let (h, mut rx) = chat_handle(&base, false, Some(agent_caller()));
    h.chat(&ask(), 16).await.unwrap();
    assert_eq!(seen.lock().unwrap().clone(), vec![SeenIdentity::default()]);
    let e = rx.try_recv().expect("the chat event was emitted");
    assert_eq!(e.get("service_key"), Some("strata-rag-chat"));
    assert_eq!(e.tenant_id().unwrap().to_string(), ROUTED_TENANT);
    assert_eq!(e.get("app_id"), Some("agent"));
}
