"""Integrity of the committed dataset fixture and KAG reference files."""

from __future__ import annotations

import json

from examples import DATA_FILE, ext_id, load_examples
from fetch_dataset import KAG_ROWS, kag_example_ids
from report import KAG_SUMMARY


def test_fixture_has_exactly_the_50_kag_questions() -> None:
    rows = [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]
    assert len(rows) == 50
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 50
    assert set(ids) == kag_example_ids(KAG_ROWS)
    for r in rows:
        assert set(r) == {
            "id",
            "question",
            "answer",
            "type",
            "level",
            "context",
            "supporting_facts",
        }


def test_every_example_has_gold_passages_and_an_answer() -> None:
    examples = load_examples()
    assert len(examples) == 50
    for ex in examples:
        assert ex.question and ex.gold_answer and ex.passages
        gold = ex.gold_external_ids()
        assert len(gold) >= 1, ex.id
        assert all(g.startswith(f"hotpotqa:{ex.id}:") for g in gold)
    # The corpus KAG ingested: one passage per non-empty context paragraph.
    assert 450 <= sum(len(ex.passages) for ex in examples) <= 500


def test_passage_ids_are_deterministic_and_unique() -> None:
    examples = load_examples()
    ids = [pid for ex in examples for pid in ex.passage_ext_ids()]
    assert len(ids) == len(set(ids))
    assert (
        ext_id("hotpotqa", "5a8b57f25542995d1e6f1371", 3) == "hotpotqa:5a8b57f25542995d1e6f1371:3"
    )


def test_kag_reference_files_are_the_promoted_baseline() -> None:
    rows = [json.loads(line) for line in KAG_ROWS.read_text().splitlines() if line.strip()]
    assert len(rows) == 250
    # text-free: no passage or answer text in the per-row evidence
    assert not any("answer" in r or "question" in r for r in rows)
    summary = json.loads(KAG_SUMMARY.read_text())
    ladder = {r["mode"]: r for r in summary["benchmarks"][0]["ladder"]}
    assert {m: ladder[m]["judge_correct"] for m in ladder} == {
        "dense": 0.78,
        "hybrid": 0.7,
        "hybrid_rerank": 0.82,
        "full_nobridge": 0.82,
        "full": 0.84,
    }
    assert [ladder[m]["recall_at_5"] for m in ladder] == [0.89, 0.9, 0.93, 0.96, 0.93]
