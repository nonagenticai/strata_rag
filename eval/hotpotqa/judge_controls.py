"""Positive / negative controls on the judge, before any judge_correct is trusted.

    STRATA_BENCH_JUDGE_PROTOCOL=vllm uv run python judge_controls.py --out runs/judge-controls.json

Every control goes through ``qa.score`` (the verbatim KAG scorer and prompt) with
``abstained=False``, so the judge is always called:

* ``positive`` - ANSWER is the question's own gold answer.          bound: >= 48/50 correct
* ``negative`` - ANSWER is the NEXT question's gold (rotate by 1).   bound: <=  2/50 correct
* ``citation_only`` - ANSWER is the bare marker ``[2]``.             bound: 0 correct

Exit status 3 when positive or negative misses its bound: the instrument is broken and no run
judged with it means anything. An unscored control (judge failure) counts against the bound.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import qa
from config import BenchConfig
from examples import DATA_FILE, Example, load_examples
from judge import Judge
from judge_vllm import make_judge

CITATION_ONLY = "[2]"


def control_answers(examples: list[Example]) -> dict[str, list[str]]:
    golds = [e.gold_answer or "" for e in examples]
    n = len(golds)
    return {
        "positive": golds,
        "negative": [golds[(i + 1) % n] for i in range(n)],
        "citation_only": [CITATION_ONLY] * n,
    }


def bounds(n: int) -> dict[str, tuple[str, int]]:
    """(comparison, threshold) on the number judged correct. n=50 gives 48 / 2 / 0."""
    return {"positive": (">=", n - 2), "negative": ("<=", 2), "citation_only": ("<=", 0)}


async def run_controls(
    judge: Judge, examples: list[Example], *, concurrency: int = 1
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max(1, min(concurrency, 2)))
    items: list[dict[str, Any]] = []

    async def one(kind: str, ex: Example, answer: str) -> dict[str, Any]:
        async with sem:
            row = await qa.score(ex, answer, judge=judge, abstained=False)
        return {
            "control": kind,
            "example_id": ex.id,
            "gold": ex.all_gold_answers(),
            "answer": answer,
            "judge_correct": row.get("judge_correct"),
            "judge_reason": row.get("judge_reason"),
        }

    for kind, answers in control_answers(examples).items():
        items += await asyncio.gather(
            *(one(kind, ex, a) for ex, a in zip(examples, answers, strict=True))
        )
        print(f"[controls] {kind} done", file=sys.stderr, flush=True)

    summary: dict[str, Any] = {}
    for kind, (op, thr) in bounds(len(examples)).items():
        rows = [i for i in items if i["control"] == kind]
        correct = sum(1 for i in rows if i["judge_correct"] == 1.0)
        unscored = sum(1 for i in rows if i["judge_correct"] is None)
        # An unscored row is a failure of the instrument, so it counts AGAINST the bound.
        passed = correct >= thr if op == ">=" else correct + unscored <= thr
        summary[kind] = {
            "n": len(rows),
            "correct": correct,
            "unscored": unscored,
            "bound": f"{op} {thr}",
            "passed": passed,
        }
    return {"summary": summary, "items": items}


async def main_async(args: argparse.Namespace) -> int:
    cfg = BenchConfig()
    judge, http = make_judge(cfg)
    try:
        result = await run_controls(judge, load_examples(args.data), concurrency=args.concurrency)
    finally:
        await http.aclose()
    out = {
        "at": datetime.now(UTC).isoformat(),
        "judge": cfg.describe(),
        "judge_counters": {
            "failures": judge.failures,
            "failed_attempts": judge.failed_attempts,
            "served_models": sorted(judge.served_models),
            "served_providers": sorted(judge.served_providers),
            "last_error": judge.last_error,
        },
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    s = result["summary"]
    if not (s["positive"]["passed"] and s["negative"]["passed"]):
        print("JUDGE CONTROLS FAILED: the instrument is broken", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="judge positive/negative controls")
    ap.add_argument("--out", type=Path, default=Path("runs/judge-controls.json"))
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--concurrency", type=int, default=1, choices=(1, 2))
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
