"""Re-score a finished run offline: no strata calls, only the judge.

    STRATA_BENCH_JUDGE_PROTOCOL=vllm uv run python rejudge.py runs/leaf-embed
    uv run python rejudge.py runs/leaf-embed --no-judge   # abstention + token metrics only

Reads ``answers.jsonl`` (the answer text) and ``results.jsonl`` (``n_citations`` and every
retrieval/latency field), recomputes ``abstained`` / ``abstain_reason`` with
``abstention.persisted`` and re-runs ``qa.score`` (the verbatim KAG scorer) with the judge, then
rewrites ``judge_correct`` / ``judge_reason`` (and the token metrics, which must not move) into
``results.jsonl``.

The first rejudge copies the original to ``results.nojudge.jsonl`` (and ``meta.json`` to
``meta.nojudge.json``); every later rejudge reads from that backup, so re-running is idempotent
and never stacks one judging on top of another. Rows that errored at ask time are kept as-is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import qa
from abstention import persisted
from config import BenchConfig
from examples import DATA_FILE, Example, load_examples
from judge import Judge
from judge_vllm import make_judge

#: Fields ``qa.score`` writes; anything else in a row comes from the original run.
TOKEN_METRICS = ("em", "cover_em", "f1", "f1_ceiling", "f1_attainment")
JUDGE_FIELDS = ("judge_correct", "judge_reason", "judge_error")
MAX_CONCURRENCY = 2  # the model is shared


def log(msg: str) -> None:
    print(f"[rejudge {datetime.now(UTC):%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def source_rows(run_dir: Path) -> list[dict[str, Any]]:
    """The un-rejudged rows: the backup if one exists, else the live file (backed up first)."""
    live, backup = run_dir / "results.jsonl", run_dir / "results.nojudge.jsonl"
    if not backup.exists():
        shutil.copy2(live, backup)
        log(f"backed up {live.name} -> {backup.name}")
    meta, meta_backup = run_dir / "meta.json", run_dir / "meta.nojudge.json"
    if meta.exists() and not meta_backup.exists():
        shutil.copy2(meta, meta_backup)
    return read_jsonl(backup)


def _moved(old: object, new: object) -> bool:
    """True when a metric changed; ``None`` (e.g. f1_attainment with a zero ceiling) is a value."""
    if isinstance(old, int | float) and isinstance(new, int | float):
        return abs(float(old) - float(new)) > 1e-9
    return old != new


async def rescore_row(
    row: dict[str, Any], ex: Example, answer: str, judge: Judge | None
) -> tuple[dict[str, Any], list[str]]:
    """The re-scored row plus the token metrics that moved (must be empty)."""
    out = {k: v for k, v in row.items() if k not in JUDGE_FIELDS}
    abstained, reason = persisted(answer, int(row.get("n_citations") or 0))
    out["abstained"], out["abstain_reason"] = abstained, reason
    scored = await qa.score(ex, answer, judge=judge, abstained=abstained, abstain_reason=reason)
    drift = [k for k in TOKEN_METRICS if k in row and k in scored and _moved(row[k], scored[k])]
    out.update(scored)
    if judge is not None and "judge_correct" not in out:
        out["judge_error"] = judge.last_error
    return out, drift


async def rejudge(
    run_dir: Path,
    *,
    judge: Judge | None,
    concurrency: int = 1,
    data: Path = DATA_FILE,
) -> dict[str, Any]:
    rows = source_rows(run_dir)
    answers = {a["example_id"]: a for a in read_jsonl(run_dir / "answers.jsonl")}
    examples = {e.id: e for e in load_examples(data)}
    sem = asyncio.Semaphore(max(1, min(concurrency, MAX_CONCURRENCY)))
    drifted: dict[str, list[str]] = {}

    async def one(i: int, row: dict[str, Any]) -> dict[str, Any]:
        if "error" in row:
            return row
        eid = row["example_id"]
        ans = answers.get(eid)
        if ans is None or "answer" not in ans:
            raise RuntimeError(f"{run_dir}: no answer text for {eid}")
        async with sem:
            new, drift = await rescore_row(row, examples[eid], str(ans["answer"]), judge)
        if drift:
            drifted[eid] = drift
        log(f"{run_dir.name} {i + 1}/{len(rows)} {eid} judge={new.get('judge_correct')}")
        return new

    new_rows = list(await asyncio.gather(*(one(i, r) for i, r in enumerate(rows))))
    write_jsonl_atomic(run_dir / "results.jsonl", new_rows)
    return {
        "n_rows": len(new_rows),
        "n_errors": sum(1 for r in new_rows if "error" in r),
        "n_scored": sum(1 for r in new_rows if isinstance(r.get("judge_correct"), int | float)),
        "n_true_abstentions": sum(1 for r in new_rows if r.get("abstained") is True),
        "n_partial_context": sum(
            1 for r in new_rows if r.get("abstain_reason") == "partial_context"
        ),
        "token_metric_drift": drifted,
    }


def update_meta(run_dir: Path, cfg: BenchConfig | None, judge: Judge | None, stats: dict) -> None:
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta["judge"] = cfg.describe() if cfg is not None else None
    meta["judge_counters"] = (
        {
            "failures": judge.failures,
            "failed_attempts": judge.failed_attempts,
            "served_models": sorted(judge.served_models),
            "served_providers": sorted(judge.served_providers),
            "last_error": judge.last_error,
        }
        if judge is not None
        else None
    )
    meta["rejudged_at"] = datetime.now(UTC).isoformat()
    meta["rejudge"] = stats
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    cfg = None if args.no_judge else BenchConfig()
    judge, http = make_judge(cfg) if cfg is not None else (None, None)
    try:
        for run_dir in args.runs:
            stats = await rejudge(run_dir, judge=judge, concurrency=args.concurrency)
            update_meta(run_dir, cfg, judge, stats)
            log(f"{run_dir}: {json.dumps(stats)}")
            if stats["token_metric_drift"]:
                log(f"{run_dir}: TOKEN METRICS MOVED - the scorer is not the one that ran")
                return 2
    finally:
        if http is not None:
            await http.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="offline re-scoring of a HotpotQA run")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--concurrency", type=int, default=1, choices=range(1, MAX_CONCURRENCY + 1))
    ap.add_argument("--no-judge", action="store_true", help="recompute abstention/token metrics")
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
