"""Run the HotpotQA n=50 benchmark against a live strata server.

    uv run python run.py --scope leaf --route-mode hybrid --out runs/leaf-hybrid

Ingests every passage of all 50 examples (the same ~500-passage corpus KAG ingested) into one
leaf store ``hotpotqa-n50``, asks each question through ``POST /api/ask``, scores it with the
copied KAG scorers + judge, and writes:

* ``results.jsonl`` - one text-free row per question (metrics, latency, counts)
* ``answers.jsonl`` - question, gold and answer text (kept apart so results stay text-free)
* ``meta.json``     - run configuration, ingest stats, judge provenance/failure counters

The store is DELETED at the end - also on error and Ctrl-C - unless ``--keep-store``: a leftover
leaf under the root changes routing for every other strata query, including ``strata eval``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import metrics
import qa
from abstention import persisted
from config import BenchConfig
from examples import DATA_FILE, Example, load_examples
from judge import Judge
from judge_vllm import make_judge
from strata_client import (
    ROUTE_MODES,
    DocIndex,
    StrataClient,
    delete_store_sync,
    predicted_ranking,
)

STORE_NAME = "hotpotqa-n50"
STORE_DESCRIPTION = "HotpotQA distractor passages (Wikipedia paragraphs) for benchmarking"
DEFAULT_BASE_URL = "http://127.0.0.1:8092"


def log(msg: str) -> None:
    print(f"[hotpotqa {datetime.now(UTC):%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="strata HotpotQA n=50 benchmark")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--scope",
        choices=("leaf", "tree"),
        default="leaf",
        help="leaf: root_id = the benchmark store (retrieval core only, comparable to KAG); "
        "tree: root_id null, routed through the whole store tree",
    )
    ap.add_argument("--route-mode", choices=ROUTE_MODES, default="hybrid")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--expand",
        action="store_true",
        help="query expansion (HyDE passage + 3 rewrites, KAG's FULL-mode technique)",
    )
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--keep-store", action="store_true", help="do not delete the store at the end")
    ap.add_argument(
        "--reuse-store",
        action="store_true",
        help=f"reuse an existing {STORE_NAME!r} store (re-POSTs passages; dedup makes it cheap)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="ask only the first N questions")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    args = ap.parse_args(argv)
    if not 1 <= args.top_k <= 50:
        ap.error("--top-k must be 1..50 (strata's MAX_TOP_K)")
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    return args


@dataclass
class RunState:
    """Shared with the sync ``finally`` so cleanup survives a dead event loop."""

    store_id: str | None = None
    created_store: bool = False
    index: DocIndex = field(default_factory=DocIndex)


def _f(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) else None


async def ingest_corpus(
    client: StrataClient, store_id: str, examples: list[Example], index: DocIndex
) -> int:
    n = 0
    for i, ex in enumerate(examples, 1):
        for pid, p in zip(ex.passage_ext_ids(), ex.passages, strict=True):
            out = await client.ingest(store_id, text=p.text, source=pid, title=p.title)
            index.add(str(out["id"]), pid, str(out.get("status") or ""))
            n += 1
        if i % 10 == 0:
            log(f"ingested {n} passages ({i}/{len(examples)} examples)")
    return n


async def score_one(
    client: StrataClient,
    judge: Judge | None,
    ex: Example,
    *,
    args: argparse.Namespace,
    root_id: str | None,
    index: DocIndex,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row: dict[str, Any] = {
        "example_id": ex.id,
        "scope": args.scope,
        "route_mode": args.route_mode,
        "question_type": ex.question_type,
    }
    answer_row: dict[str, Any] = {
        "example_id": ex.id,
        "scope": args.scope,
        "route_mode": args.route_mode,
        "question": ex.question,
        "gold": ex.all_gold_answers(),
    }
    t0 = time.perf_counter()
    try:
        resp = await client.ask(
            ex.question,
            top_k=args.top_k,
            route_mode=args.route_mode,
            root_id=root_id,
            max_tokens=args.max_tokens,
            expand=args.expand,
        )
    except Exception as e:  # one failed ask is a counted error row, never a dropped row
        row["error"] = f"{type(e).__name__}: {e}"[:500]
        row["latency_ms_e2e"] = (time.perf_counter() - t0) * 1000.0
        answer_row["error"] = row["error"]
        return row, answer_row
    row["latency_ms_e2e"] = (time.perf_counter() - t0) * 1000.0

    answer = str(resp.get("answer") or "")
    results = list(resp.get("results") or [])
    timings = resp.get("timings_ms") or {}
    citations = list(resp.get("citations") or [])
    gold = ex.gold_external_ids()
    ranking = predicted_ranking(results, index, gold)
    abstained, reason = persisted(answer, len(citations))

    routing = resp.get("routing") if isinstance(resp.get("routing"), dict) else None
    expansion = resp.get("expansion") if isinstance(resp.get("expansion"), dict) else None
    row.update(
        {
            "n_gold": len(gold),
            "recall_at_5": metrics.recall_at_k(ranking.passage_ids, gold, 5),
            "recall_at_10": metrics.recall_at_k(ranking.passage_ids, gold, 10),
            "ndcg_at_10": metrics.ndcg_at_k(ranking.passage_ids, gold, 10),
            "mrr": metrics.mrr(ranking.passage_ids, gold),
            "n_results": len(results),
            "n_docs_retrieved": len(ranking.passage_ids),
            "dedup_collisions_top5": ranking.collisions_top5,
            "collision_gold_credits_top5": ranking.collision_gold_credits_top5,
            "unmapped_docs": ranking.unmapped_docs,
            "abstained": abstained,
            "abstain_reason": reason,
            "n_citations": len(citations),
            # /api/ask returns no routing block (src/ask.rs AskResponse); null = not observable.
            "leaves_used": len(routing.get("leaves") or []) if routing else None,
            "laya_consulted": routing.get("laya_consulted") if routing else None,
            "leaves_in_results": len({tuple(r.get("store_path") or []) for r in results}),
            "latency_ms_query": _f(timings.get("total")),
            "latency_ms_generate": _f(timings.get("generate")),
            # null when not requested; a fallback string means this row ran UNexpanded
            "expansion_fallback": expansion.get("fallback") if expansion else None,
            "latency_ms_expand": _f(expansion.get("ms")) if expansion else None,
            "timings_ms": {
                k: _f(timings.get(k))
                for k in ("embed", "route", "search", "rerank", "total", "generate")
            },
        }
    )
    try:
        row.update(
            await qa.score(ex, answer, judge=judge, abstained=abstained, abstain_reason=reason)
        )
    except Exception as e:
        row["error"] = f"score: {type(e).__name__}: {e}"[:500]
    if judge is not None and "judge_correct" not in row:
        row["judge_error"] = judge.last_error
    answer_row["answer"] = answer
    return row, answer_row


async def preflight_judge(judge: Judge) -> None:
    """One cheap call before any ingest: a 401/403 or dead router aborts the run up front."""
    await judge.judge_json_strict(
        'Reply with ONLY this JSON: {"ok": true}', "ping", require=("ok",)
    )


async def run_async(args: argparse.Namespace, state: RunState) -> dict[str, Any]:
    examples = load_examples(args.data)
    asked = examples[: args.limit] if args.limit > 0 else examples
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = BenchConfig()
    judge: Judge | None = None
    judge_http = None
    if not args.no_judge:
        judge, judge_http = make_judge(cfg)

    meta: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "scope": args.scope,
        "route_mode": args.route_mode,
        "expand": args.expand,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "n_examples_corpus": len(examples),
        "n_examples_asked": len(asked),
        "judge": cfg.describe() if judge else None,
    }
    try:
        async with StrataClient(args.base_url) as client:
            meta["health"] = await client.health()
            if judge is not None:
                await preflight_judge(judge)
                log("judge preflight ok")

            existing = await client.find_store(STORE_NAME)
            if existing and not args.reuse_store:
                raise RuntimeError(
                    f"store {STORE_NAME!r} ({existing['id']}) already exists; pass --reuse-store "
                    "or delete it (DELETE /api/stores/<id>) first"
                )
            if existing:
                state.store_id = str(existing["id"])
                log(f"reusing store {state.store_id}")
            else:
                created = await client.create_leaf(STORE_NAME, STORE_DESCRIPTION)
                state.store_id = str(created["id"])
                state.created_store = True
                log(f"created store {state.store_id} (parent {created.get('parent_id')})")
            meta["store_id"] = state.store_id

            t0 = time.perf_counter()
            n_passages = await ingest_corpus(client, state.store_id, examples, state.index)
            meta["ingest"] = {
                "n_passages": n_passages,
                "n_documents": len(state.index.sources),
                "created": state.index.created,
                "existing": state.index.existing,
                "collision_docs": state.index.collision_docs(),
                "collided_passages": state.index.collided_passages(),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            log(f"ingest: {meta['ingest']}")

            root_id = state.store_id if args.scope == "leaf" else None
            sem = asyncio.Semaphore(args.concurrency)

            async def one(ex: Example) -> tuple[dict[str, Any], dict[str, Any]]:
                async with sem:
                    return await score_one(
                        client, judge, ex, args=args, root_id=root_id, index=state.index
                    )

            res_path, ans_path = args.out / "results.jsonl", args.out / "answers.jsonl"
            n_err = n_unexpanded = 0
            with (
                res_path.open("w", encoding="utf-8") as rf,
                ans_path.open("w", encoding="utf-8") as af,
            ):
                for i, fut in enumerate(asyncio.as_completed([one(ex) for ex in asked]), 1):
                    row, ans = await fut
                    n_err += "error" in row
                    n_unexpanded += row.get("expansion_fallback") is not None
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    af.write(json.dumps(ans, ensure_ascii=False) + "\n")
                    rf.flush()
                    af.flush()
                    log(
                        f"{i}/{len(asked)} {row['example_id']} r@5={row.get('recall_at_5')} "
                        f"judge={row.get('judge_correct')} err={'error' in row}"
                    )
            meta["n_rows"] = len(asked)
            meta["n_errors"] = n_err
            # rows that asked for expansion but ran without it (vLLM error / unparseable reply)
            meta["n_expansion_fallbacks"] = n_unexpanded
    finally:
        if judge is not None:
            meta["judge_counters"] = {
                "failures": judge.failures,
                "failed_attempts": judge.failed_attempts,
                "served_models": sorted(judge.served_models),
                "served_providers": sorted(judge.served_providers),
                "last_error": judge.last_error,
            }
        if judge_http is not None:
            await judge_http.aclose()
        meta["finished_at"] = datetime.now(UTC).isoformat()
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = RunState()
    rc = 1
    try:
        meta = asyncio.run(run_async(args, state))
        log(f"done: {meta.get('n_rows')} rows, {meta.get('n_errors')} errors -> {args.out}")
        rc = 0
    except KeyboardInterrupt:
        log("interrupted")
        rc = 130
    finally:
        if state.store_id and not args.keep_store:
            try:
                gone = delete_store_sync(args.base_url, state.store_id)
                log(f"deleted store {state.store_id}" if gone else "store already gone")
            except Exception as e:
                log(f"FAILED to delete store {state.store_id}: {e} - delete it by hand")
                rc = rc or 1
        elif state.store_id:
            log(f"kept store {state.store_id} (--keep-store)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
