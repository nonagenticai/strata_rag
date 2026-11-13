"""rejudge.py: offline re-scoring rewrites judge + abstention fields, backs up the original."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import rejudge
from examples import load_examples


class StubJudge:
    """Says correct=true iff the gold string is in the answer; counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_error: str | None = None

    async def judge_json(self, system: str, user: str, **_: object) -> dict[str, Any]:
        self.calls += 1
        gold = user.split("GOLD ANSWER(S):\n", 1)[1].split("\n", 1)[0]
        answer = user.split("\n\nANSWER:\n", 1)[1]
        return {"correct": gold.lower() in answer.lower(), "reason": "stub"}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_rejudge_rewrites_rows_and_keeps_a_backup(tmp_path: Path) -> None:
    ex = load_examples()[:4]
    answers = [
        f"It is {ex[0].gold_answer} [1].",  # correct
        "Something else entirely [2].",  # wrong
        "The sources do not mention it, but X is true [1].",  # partial_context -> judged
        "I don't know.",  # true abstention -> 0.0 without a judge call
    ]
    results = [
        {"example_id": e.id, "n_citations": 0 if i == 3 else 1, "recall_at_5": 0.5}
        for i, e in enumerate(ex)
    ]
    # the old runner's shape: a partial row persisted as abstained=True
    results[2].update(abstained=True, abstain_reason="partial_context")
    _write(tmp_path / "results.jsonl", results)
    _write(
        tmp_path / "answers.jsonl",
        [{"example_id": e.id, "answer": a} for e, a in zip(ex, answers, strict=True)],
    )
    judge = StubJudge()
    stats = asyncio.run(rejudge.rejudge(tmp_path, judge=judge))  # type: ignore[arg-type]

    out = rejudge.read_jsonl(tmp_path / "results.jsonl")
    assert [r["judge_correct"] for r in out] == [1.0, 0.0, 0.0, 0.0]
    assert judge.calls == 3
    assert out[2]["abstained"] is False and out[2]["abstain_reason"] == "partial_context"
    assert out[3]["abstained"] is True and "judge_reason" not in out[3]
    assert all(r["recall_at_5"] == 0.5 for r in out)
    assert rejudge.read_jsonl(tmp_path / "results.nojudge.jsonl") == results
    assert stats["n_scored"] == 4 and stats["n_true_abstentions"] == 1
    assert stats["n_partial_context"] == 1 and stats["token_metric_drift"] == {}

    # a second rejudge reads the backup, not its own output
    asyncio.run(rejudge.rejudge(tmp_path, judge=StubJudge()))  # type: ignore[arg-type]
    assert rejudge.read_jsonl(tmp_path / "results.nojudge.jsonl") == results
    assert rejudge.read_jsonl(tmp_path / "results.jsonl") == out


def test_error_rows_pass_through_untouched(tmp_path: Path) -> None:
    ex = load_examples()[:1]
    row = {"example_id": ex[0].id, "error": "boom"}
    _write(tmp_path / "results.jsonl", [row])
    _write(tmp_path / "answers.jsonl", [{"example_id": ex[0].id, "error": "boom"}])
    judge = StubJudge()
    asyncio.run(rejudge.rejudge(tmp_path, judge=judge))  # type: ignore[arg-type]
    assert rejudge.read_jsonl(tmp_path / "results.jsonl") == [row]
    assert judge.calls == 0


def test_moved_treats_none_as_a_value() -> None:
    assert not rejudge._moved(None, None)  # pyright: ignore[reportPrivateUsage]
    assert rejudge._moved(None, 0.5)  # pyright: ignore[reportPrivateUsage]
    assert not rejudge._moved(1, 1.0)  # pyright: ignore[reportPrivateUsage]
