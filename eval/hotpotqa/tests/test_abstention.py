"""strata answer text -> KAG's three-state abstention contract."""

from __future__ import annotations

import pytest

import metrics
from abstention import classify, is_true_abstention, persisted


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   \n",
        "I don't know.",
        "I do not know based on the provided sources.",
        "The sources do not contain information about the director's birthplace.",
        "The provided sources don't mention when the band was formed.",
        "I cannot determine the answer from the sources.",
        "There is not enough information in the sources to answer this question.",
        "The release year is not mentioned in the sources.",
    ],
)
def test_true_abstentions(answer: str) -> None:
    assert classify(answer, 0) == (True, None)
    assert is_true_abstention(answer, 0)


def test_partial_answer_with_citations_is_scored_on_content() -> None:
    answer = (
        "Ed Wood was born in Poughkeepsie [1], but the sources do not say where Scott was born."
    )
    assert classify(answer, 1) == (True, metrics.PARTIAL_ABSTAIN_REASON)
    assert not is_true_abstention(answer, 1)


@pytest.mark.parametrize(
    "answer",
    [
        "Yes, both Scott Derrickson and Ed Wood were American [1][2].",
        "The answer is Chief of Protocol [3].",
        "Arthur's Magazine was started first, in 1844 [1].",
        "He did not win the award; it went to Jones [2].",  # negation that is not abstention
    ],
)
def test_real_answers_are_not_abstentions(answer: str) -> None:
    assert classify(answer, 1) == (False, None)
    assert classify(answer, 0) == (False, None)


def test_true_abstention_short_circuits_judge_like_kag() -> None:
    import asyncio

    import qa
    from examples import Example

    ex = Example(id="x", benchmark="hotpotqa", question="q?", gold_answer="Paris")

    class Boom:
        async def judge_json(self, *a: object, **k: object) -> dict[str, object]:
            raise AssertionError("judge must not be called for a true abstention")

    ab, reason = classify("I don't know.", 0)
    row = asyncio.run(
        qa.score(ex, "I don't know.", judge=Boom(), abstained=ab, abstain_reason=reason)  # type: ignore[arg-type]
    )
    assert row["judge_correct"] == 0.0


_PARTIAL = "Ed Wood was born in Poughkeepsie [1], but the sources do not say where Scott was born."


def test_persisted_partial_context_is_not_abstained() -> None:
    assert persisted(_PARTIAL, 1) == (False, metrics.PARTIAL_ABSTAIN_REASON)
    assert persisted("I don't know.", 0) == (True, None)
    assert persisted("Paris [1].", 1) == (False, None)


def test_results_row_for_partial_context_matches_is_true_abstention() -> None:
    """The row run.py writes: a partial answer is NOT abstained (README, KAG summary.py).

    The old runner wrote ``abstained=True`` for ``partial_context`` rows, which contradicts
    ``metrics.row_is_true_abstention`` on the very same row.
    """
    import asyncio
    from argparse import Namespace

    import run
    from examples import Example
    from strata_client import DocIndex

    class FakeStrata:
        async def ask(self, *a: object, **k: object) -> dict[str, object]:
            return {"answer": _PARTIAL, "results": [], "citations": [{"n": 1}], "timings_ms": {}}

    ex = Example(id="x", benchmark="hotpotqa", question="q?", gold_answer="Poughkeepsie")
    args = Namespace(scope="leaf", route_mode="hybrid", top_k=10, max_tokens=64, expand=False)
    row, _ = asyncio.run(
        run.score_one(FakeStrata(), None, ex, args=args, root_id=None, index=DocIndex())  # type: ignore[arg-type]
    )
    assert row["abstain_reason"] == metrics.PARTIAL_ABSTAIN_REASON
    assert row["abstained"] is False
    assert row["abstained"] == metrics.row_is_true_abstention(row)
