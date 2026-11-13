"""Write report.md: strata run(s) next to KAG's promoted HotpotQA n=50 baseline.

    uv run python report.py runs/leaf-hybrid runs/tree-hybrid --out runs/report.md

KAG reference numbers (recall@5, cover_em, judge_correct, abstain_rate, n_scored, n_errors) come
from ``data/kag_n50_20260902.summary.json`` - the PROMOTED baseline. KAG latency and the paired
per-question comparison come from ``data/kag_n50_20260902.jsonl``, KAG's only retained per-row
file, which is from a LATER re-run (35268554903) of the same 50 questions, not the promoted run
itself. Its recall@5 tracks the promoted values, but its judge_correct differs by up to 4 rows
per mode (judge noise), so the paired table prints KAG's mean over the paired questions rather
than the promoted number. Every mean is printed with its denominator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import metrics

HERE = Path(__file__).resolve().parent
KAG_SUMMARY = HERE / "data" / "kag_n50_20260902.summary.json"
KAG_ROWS = HERE / "data" / "kag_n50_20260902.jsonl"
KAG_MODES = ("dense", "hybrid", "hybrid_rerank", "full_nobridge", "full")
PAIRED_AGAINST = ("full_nobridge", "dense")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _nums(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if isinstance(r.get(key), int | float)]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean_cell(rows: list[dict[str, Any]], key: str, n: int) -> str:
    """Mean with its denominator whenever it is not the full row count."""
    vals = _nums(rows, key)
    if not vals:
        return "-"
    m = sum(vals) / len(vals)
    return f"{m:.3f}" if len(vals) == n else f"{m:.3f} (n={len(vals)})"


def ms(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f}"


def strata_summary(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "results.jsonl")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    lat = _nums(ok, "latency_ms_e2e")
    return {
        "label": (
            f"strata {rows[0]['scope']}/{rows[0]['route_mode']}"
            + ("+expand" if meta.get("expand") else "")
            if rows
            else run_dir.name
        ),
        "n": len(rows),
        "n_scored": len(_nums(ok, "judge_correct")),
        "n_errors": len(rows) - n,
        "recall_at_5": mean_cell(ok, "recall_at_5", n),
        "cover_em": mean_cell(ok, "cover_em", n),
        "judge_correct": mean_cell(ok, "judge_correct", n),
        "abstain_rate": (
            f"{sum(1 for r in ok if metrics.row_is_true_abstention(r)) / n:.3f}" if n else "-"
        ),
        "p50": percentile(lat, 50),
        "p95": percentile(lat, 95),
        "p50_query": percentile(_nums(ok, "latency_ms_query"), 50),
        "collisions": sum(int(r.get("collision_gold_credits_top5") or 0) for r in ok),
        "ingest": meta.get("ingest"),
        "judge_counters": meta.get("judge_counters"),
        "rows": rows,
    }


def kag_summaries(summary_path: Path, rows_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    bench = next(b for b in summary["benchmarks"] if b["benchmark"] == "hotpotqa")
    per_row = read_jsonl(rows_path)
    out = []
    for rung in bench["ladder"]:
        mode = rung["mode"]
        lat = _nums(
            (r for r in per_row if r.get("mode") == mode and "error" not in r), "latency_ms"
        )

        def fmt(x: Any) -> str:
            return "-" if x is None else f"{float(x):.3f}"

        out.append(
            {
                "label": f"KAG {mode}",
                "n": rung["n"],
                "n_scored": rung["n_scored"],
                "n_errors": rung["n_errors"],
                "recall_at_5": fmt(rung.get("recall_at_5")),
                "cover_em": fmt(rung.get("cover_em")),
                "judge_correct": fmt(rung.get("judge_correct")),
                "abstain_rate": fmt(rung.get("abstain_rate")),
                "p50": percentile(lat, 50),
                "p95": percentile(lat, 95),
                "p50_query": None,
            }
        )
    return out


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for k successes of n at p=0.5."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(
    strata_rows: list[dict[str, Any]], kag_rows: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    s = {
        r["example_id"]: float(r["judge_correct"])
        for r in strata_rows
        if "error" not in r and isinstance(r.get("judge_correct"), int | float)
    }
    k = {
        r["example_id"]: float(r["judge_correct"])
        for r in kag_rows
        if r.get("mode") == mode
        and "error" not in r
        and isinstance(r.get("judge_correct"), int | float)
    }
    both = sorted(s.keys() & k.keys())
    wins = sum(1 for e in both if s[e] > k[e])
    losses = sum(1 for e in both if s[e] < k[e])
    return {
        "mode": mode,
        "n_paired": len(both),
        "unpaired_strata": len(s.keys() - k.keys()),
        "unpaired_kag": len(k.keys() - s.keys()),
        "wins": wins,
        "losses": losses,
        "ties": len(both) - wins - losses,
        "strata_mean": sum(s[e] for e in both) / len(both) if both else None,
        "kag_mean": sum(k[e] for e in both) / len(both) if both else None,
        "sign_p": binom_two_sided(wins, wins + losses),
    }


def render(strata: list[dict[str, Any]], kag: list[dict[str, Any]], kag_rows: list[dict]) -> str:
    lines = [
        "# strata vs KAG - HotpotQA distractor n=50",
        "",
        "Same 50 questions, same scorers (copied from KAG), same pinned judge model `local-llm`.",
        "Means are over non-error rows; a mean over fewer rows than `n - n_errors` shows its own",
        "`(n=...)`. Latency is end-to-end wall clock per question, generation included.",
        "",
        "| system | n | n_scored | n_errors | recall@5 | cover_em | judge_correct | abstain_rate "
        "| p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in [*strata, *kag]:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['n_scored']} | {r['n_errors']} | {r['recall_at_5']} "
            f"| {r['cover_em']} | {r['judge_correct']} | {r['abstain_rate']} | {ms(r['p50'])} "
            f"| {ms(r['p95'])} |"
        )
    lines += [
        "",
        "KAG rows: metrics from the promoted summary `n50-20260902`; KAG latency (median / p95 of",
        "`latency_ms`, includes HyDE/graph + generation, measured on the cluster) from the per-row",
        "re-run 35268554903.",
        "",
    ]
    for s in strata:
        lines.append(f"## {s['label']}")
        lines.append("")
        lines.append(f"- query-only latency p50: {ms(s['p50_query'])} ms (timings_ms.total)")
        lines.append(f"- ingest: {json.dumps(s['ingest'])}")
        lines.append(
            "- top-5 gold credits that came only through a content-dedup collision: "
            f"{s['collisions']}"
        )
        if s["judge_counters"]:
            lines.append(f"- judge counters: {json.dumps(s['judge_counters'])}")
        lines += [
            "",
            "Paired judge_correct (strata vs KAG mode, questions both scored):",
            "",
            "| vs KAG | n_paired | strata wins | KAG wins | ties | strata mean | KAG mean "
            "| sign-test p | unpaired (strata/KAG) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for mode in PAIRED_AGAINST:
            p = paired(s["rows"], kag_rows, mode)
            sm = "-" if p["strata_mean"] is None else f"{p['strata_mean']:.3f}"
            km = "-" if p["kag_mean"] is None else f"{p['kag_mean']:.3f}"
            lines.append(
                f"| {mode} | {p['n_paired']} | {p['wins']} | {p['losses']} | {p['ties']} | {sm} "
                f"| {km} | {p['sign_p']:.3f} | {p['unpaired_strata']}/{p['unpaired_kag']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="strata vs KAG HotpotQA report")
    ap.add_argument("runs", nargs="+", type=Path, help="run directories written by run.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kag-summary", type=Path, default=KAG_SUMMARY)
    ap.add_argument("--kag-rows", type=Path, default=KAG_ROWS)
    args = ap.parse_args(argv)
    strata = [strata_summary(d) for d in args.runs]
    kag_rows = read_jsonl(args.kag_rows)
    text = render(strata, kag_summaries(args.kag_summary, args.kag_rows), kag_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
