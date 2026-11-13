# HotpotQA n=50 benchmark for strata

Runs the same 50 HotpotQA questions KAG's promoted baseline (`n50-20260902`) was scored on,
through strata's `POST /api/ask`, with KAG's scorers and KAG's pinned judge, copied into
this directory. Nothing is imported from the kag repo.

## Files

| file | what |
|---|---|
| `data/n50.jsonl` | the 50 raw rows (HF `hotpotqa/hotpot_qa`, `distractor`, `validation`, its first 50 rows): 492 non-empty passages, 2 gold titles per question |
| `data/kag_n50_20260902.summary.json` | KAG's PROMOTED summary: the reference numbers |
| `data/kag_n50_20260902.jsonl` | KAG's per-row, text-free scores (250 rows = 5 modes x 50) for latency and the paired comparison |
| `fetch_dataset.py` | rebuilds `data/n50.jsonl` by streaming the split (`uv run --extra fetch python fetch_dataset.py`) |
| `metrics.py`, `qa.py`, `judge.py` | copied from kag `eval/` (headers give the sha); only the imports changed |
| `examples.py`, `config.py` | trimmed copies of kag `benchmarks/{common,multihop}.py` and the judge part of `config.py` |
| `strata_client.py` | httpx client and the `document_id` to passage-id mapping |
| `abstention.py` | strata answer text mapped onto KAG's abstention contract |
| `judge_vllm.py` | OpenAI-vLLM judge transport (`STRATA_BENCH_JUDGE_PROTOCOL=vllm`): a `Judge` subclass, so `judge.py` stays verbatim |
| `judge_controls.py` | positive / negative / citation-only controls on the judge, written to `runs/judge-controls.json` |
| `rejudge.py` | offline re-scoring of a finished run from `answers.jsonl` (no strata calls) |
| `run.py` / `report.py` | runner and report |

## Reference numbers (KAG, promoted)

| KAG mode | recall@5 | judge_correct |
|---|---|---|
| dense | 0.89 | 0.78 |
| hybrid | 0.90 | 0.70 |
| hybrid_rerank (KAG's shipped default) | 0.93 | 0.82 |
| full_nobridge | 0.96 | 0.82 |
| full | 0.93 | 0.84 |

## Running

The runner needs a running strata (default `http://127.0.0.1:8092`) with vLLM configured, so
that `/api/ask` can generate. It also needs a judge licence token.

```bash
cd eval/hotpotqa
uv sync

# judge via the public edge
export STRATA_BENCH_JUDGE_URL=https://api.nonagentic.ai/inference-router
export STRATA_BENCH_JUDGE_TOKEN_FILE=~/.config/strata-rag/licenses.env   # strata's format
export STRATA_BENCH_JUDGE_TOKEN_KEY=VLLM_LICENSE_TOKEN                   # key inside the file

uv run python run.py --scope leaf --route-mode hybrid --out runs/leaf-hybrid
uv run python run.py --scope tree --route-mode hybrid --out runs/tree-hybrid
uv run python report.py runs/leaf-hybrid runs/tree-hybrid --out runs/report.md
```

`--expand` sends `expand: true` (HyDE passage + 3 rewrites, see `src/expand.rs`). Rows that ran
unexpanded carry `expansion_fallback`, and `meta.json` counts them in `n_expansion_fallbacks`.
`--limit N` asks only the first N questions, but the full 492-passage corpus is always ingested.
`--no-judge` skips the judge. `--concurrency` defaults to 1, so latency is not inflated by
queueing. `--keep-store` and `--reuse-store` are covered below.

### Judge transport

`STRATA_BENCH_JUDGE_PROTOCOL=router` (default) calls inference-router's `/api/chat/completions`,
as KAG did. strata's licence token is for the `vllm` app only, so the router answers it with
403 `app_route_mismatch`. `STRATA_BENCH_JUDGE_PROTOCOL=vllm` calls the same instrument directly:
`POST https://api.nonagentic.ai/vllm/v1/chat/completions` (the default URL for that protocol) with
`model: local-llm` and `chat_template_kwargs: {enable_thinking: false}`. That is exactly what the
router's local provider forwards (`force_chat_template_kwargs`, reasoning off by default),
with the same prompts, `max_tokens` 900 and `temperature` 0.0. The served model is recorded from
the reply's `model` field.

```bash
export STRATA_BENCH_JUDGE_PROTOCOL=vllm
uv run python judge_controls.py --out runs/judge-controls.json   # exit 3 = instrument broken
uv run python rejudge.py runs/leaf-embed runs/tree-hybrid        # --concurrency 1 (max 2)
```

`rejudge.py` backs up `results.jsonl` to `results.nojudge.jsonl` (and `meta.json` to
`meta.nojudge.json`) the first time, and always re-scores from that backup, so it is idempotent.
It recomputes `abstained` / `abstain_reason` from the answer text and `n_citations`, re-runs
`qa.score`, and exits 2 if any token metric moved (which would mean a different scorer).

The runner makes one judge call before it ingests anything. If that call fails (for example a
401 or 403, or a router that cannot be reached), the run stops early instead of producing 50
unjudged rows.

### Store lifecycle

Strata allows exactly one root store. The runner creates the leaf `hotpotqa-n50` under the
existing root router, or makes it the root if the server is empty. It ingests one document per
passage, with `source = hotpotqa:<example_id>:<passage_idx>` and `title` = the Wikipedia title.
At the end the store is deleted with `DELETE /api/stores/{id}`, which cascades to its documents
and chunks. The delete also happens on an error or Ctrl-C. A leftover leaf would change routing
for every other query, including `strata eval`. Pass `--keep-store` to keep the store. If a
`hotpotqa-n50` store already exists, the run refuses to start unless you pass `--reuse-store`.
Re-posting into a reused store is cheap, because strata dedups by content hash and answers
`existing`.

### Outputs

- `results.jsonl`: one text-free row per question. Fields: `recall_at_5/10`, `ndcg_at_10`,
  `mrr`, `em`, `f1`, `cover_em`, `judge_correct`, `judge_reason`, `abstained`, `abstain_reason`,
  `latency_ms_query` (= `timings_ms.total`), `latency_ms_generate`, `latency_ms_e2e` (wall clock),
  `n_citations`, `leaves_used`, `laya_consulted`, `leaves_in_results`, and the dedup counters.
  A failed ask becomes a row with `error`; it is never dropped.
- `answers.jsonl`: the question, gold and answer text.
- `meta.json`: config, ingest stats, and judge provenance (`served_models`, `failures`).

`leaves_used` and `laya_consulted` are always `null` today. `/api/ask` returns no `routing` block
(see `AskResponse` in `src/ask.rs`). `leaves_in_results` (the number of distinct `store_path`s
among the results) is the closest observable value.

## What each scope measures

- **leaf** (`root_id` = the benchmark store): routing starts at the leaf, so `--route-mode` has
  no effect. This measures strata's retrieval core (dense + lexical, RRF, rerank) and its
  generation over exactly the HotpotQA corpus. This is the scope to compare with KAG.
- **tree** (`root_id` = null): the question is routed from the root through the whole store
  tree (the books plus `hotpotqa-n50`) with the chosen `--route-mode` (`embed`, `laya` or
  `hybrid`; strata has no `flat`). This also tests whether routing finds the right leaf, so a
  routing miss shows up as a recall drop.

## Scoring rules

- **Gold**: the passages whose title is in `supporting_facts` (2 per question).
- **recall@k**: the predicted ranking is `results[]` in rank order, deduplicated by
  `document_id` (a long passage can be several chunks), with each document mapped back to the
  passage id(s) it was ingested as. KAG's `metrics.recall_at_k` then scores the top k
  *distinct documents*. A document that strata does not know (from another store) keeps its
  rank as a non-gold placeholder.
- **Dedup collisions**: strata dedups identical normalised text within a store. In this
  corpus, one paragraph appears under two questions, so two passage ids share one document. That
  document counts as whichever of its passage ids is gold for the question being asked. It still
  fills only one rank and is credited only once. Each row carries `dedup_collisions_top5` and
  `collision_gold_credits_top5`. The second counter is the number of top-5 gold credits that strict
  first-source identity would have denied. KAG stored each copy as a separate document, so
  retrieving the other question's copy counted as a miss there. This favours strata slightly,
  and the counter shows by how much.
- **Abstention**: strata has no abstain flag. KAG's `metrics.py` has no phrase logic, only the
  three-state contract (`is_true_abstention`, `partial_context`). `abstention.py` maps strata's
  text onto that contract:
  - An empty answer, or an "I don't know"-style declaration with **zero citations**, is a true
    abstention. `judge_correct` is set to 0.0 without calling the judge, as KAG did.
  - A declaration with **at least one citation** is `partial_context`. It is written as
    `abstained: false, abstain_reason: "partial_context"` (`abstention.persisted`, i.e.
    `metrics.is_true_abstention`), so `abstained` in `results.jsonl` always means a true
    abstention, and the judge grades its content.

  `abstain_rate` counts true abstentions over the non-error rows, matching KAG's `summary.py`.
- **EM / F1 / cover_em / judge_correct**: KAG's `qa.score`, unchanged. Citation markers `[n]`
  are stripped before the token metrics, which is correct for strata as well, because strata
  emits the same `[n]` form.

## Comparability caveats

1. **Self-grading.** The judge model is `local-llm`, and strata generates with the same
   `local-llm` (the local vLLM). The model therefore grades its own answers. KAG's baseline was
   self-graded in exactly the same way, because its `deepseek-v4-flash` request name was served
   by `local-llm`. The judge is also noisy: in KAG's three runs, `hybrid` swung by 5 rows.
2. **Corpus.** KAG ingested the same ~500 passages, but into a shared tenant that also held
   residue from other runs (the duplicate-pool finding). Strata's leaf holds only these 492
   passages.
3. **Latency is not like-for-like.** KAG's `latency_ms` includes HyDE, graph augmentation and
   generation through the cluster edge. Strata runs locally against remote embeddings and remote
   vLLM. Compare latency only as an order of magnitude.
4. **Per-row KAG file.** The paired comparison and KAG latency use `kag_n50_20260902.jsonl`. That
   file is from KAG run 35268554903, a re-run of the same 50 questions and not the promoted run
   itself (whose per-row file was not kept). Its recall@5 is close to the promoted values
   (0.867/0.90/0.93/0.96/0.94). Its judge_correct is not: dense 0.735 (n=49), hybrid 0.78,
   hybrid_rerank 0.84, full_nobridge 0.84 and full 0.80, against the promoted
   0.78/0.70/0.82/0.82/0.84. That gap is judge noise of up to 4 rows per mode. The paired table
   therefore prints the KAG mean over the paired questions next to the strata mean. One `dense`
   row errored, so pairing against `dense` covers at most 49 questions.
5. **Retrieval depth.** Strata answers from `--top-k 10` chunks. KAG used `top_k` 8 over a
   50-candidate pool. recall@5 is computed on distinct documents in both systems.

## Tests

```bash
uv run pytest && uv run ruff check && uv run ruff format --check && uv run basedpyright
```

The tests run offline. They cover fixture integrity, the recall mapping (including dedup
collisions), abstention, and store deletion on error or Ctrl-C (strata is mocked with respx).
