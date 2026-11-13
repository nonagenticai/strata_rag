from __future__ import annotations

import json
from pathlib import Path

from report import (
    KAG_ROWS,
    KAG_SUMMARY,
    binom_two_sided,
    kag_summaries,
    paired,
    percentile,
    read_jsonl,
    strata_summary,
)


def test_percentile_linear() -> None:
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0], 95) == 10.0


def test_kag_rows_match_promoted_summary_and_carry_latency() -> None:
    rows = {r["label"]: r for r in kag_summaries(KAG_SUMMARY, KAG_ROWS)}
    assert rows["KAG full"]["judge_correct"] == "0.840"
    assert rows["KAG dense"]["recall_at_5"] == "0.890"
    assert all(r["p50"] and r["p95"] and r["p95"] >= r["p50"] for r in rows.values())


def test_paired_wins_losses_ties_and_unpaired() -> None:
    kag = read_jsonl(KAG_ROWS)
    dense = {r["example_id"]: r.get("judge_correct") for r in kag if r["mode"] == "dense"}
    ids = sorted(dense)
    strata = [{"example_id": e, "judge_correct": 1.0} for e in ids[:10]]
    strata.append({"example_id": ids[10], "error": "boom"})
    p = paired(strata, kag, "dense")
    scored = [e for e in ids[:10] if isinstance(dense[e], float)]
    assert p["n_paired"] == len(scored)
    assert p["wins"] == sum(1 for e in scored if dense[e] == 0.0)
    assert p["losses"] == 0
    assert p["unpaired_kag"] == 50 - len(scored) - (
        0 if isinstance(dense.get("5ae2070a5542994d89d5b313"), float) else 1
    )


def test_sign_test() -> None:
    assert binom_two_sided(0, 0) == 1.0
    assert abs(binom_two_sided(0, 5) - 0.0625) < 1e-9
    assert binom_two_sided(3, 6) == 1.0


def test_expanded_run_is_labelled_apart(tmp_path: Path) -> None:
    row = {"example_id": "x", "scope": "leaf", "route_mode": "embed", "latency_ms_e2e": 1.0}
    for name, expand in (("plain", False), ("hyde", True)):
        d = tmp_path / name
        d.mkdir()
        (d / "results.jsonl").write_text(json.dumps(row) + "\n")
        (d / "meta.json").write_text(json.dumps({"expand": expand}))
    assert strata_summary(tmp_path / "plain")["label"] == "strata leaf/embed"
    assert strata_summary(tmp_path / "hyde")["label"] == "strata leaf/embed+expand"
