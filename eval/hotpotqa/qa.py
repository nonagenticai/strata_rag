# copied from kag@d653afba14d3f3c82ef84dea1a88f226d20e5726
# eval/scorers/qa.py, unchanged except imports
"""Short-answer QA scoring for the multi-hop benchmarks.

KAG synthesizes prose, so strict EM under-credits a correct-but-verbose
answer (verified in smoke: a correct comparison answer scored EM=0). We
therefore report EM + token-F1 + cover-EM (pure) AND an LLM correctness
judge that asks whether the gold answer is entailed by KAG's answer — the
fair metric for generative RAG on extractive QA.

token-F1 is additionally emitted with `f1_ceiling` (the best f1 an answer of
this length could score against this gold) and `f1_attainment` (f1/ceiling).
A prose answer against a 1-3 token gold is bounded far below any published
answer-F1 anchor, so the raw number alone invites a false regression verdict.
"""

from __future__ import annotations

from typing import Any

import metrics
from examples import Example
from judge import Judge

# ⛔ The abstention clause below is NOT a style choice; it is the judge's half of
# the three-state abstention contract in ``eval/metrics.py`` (the
# ``PARTIAL_ABSTAIN_REASON`` / ``is_true_abstention`` block).
#
# It used to read, in full: *"If the ANSWER abstains or says it lacks information,
# correct=false."* That blanket rule CONTRADICTED this harness's own rule one
# module over. ``metrics.PARTIAL_ABSTAIN_REASON`` / ``metrics.is_true_abstention``
# say a ``partial_context`` row "carries a real, citable answer that must be
# judged on its content like any other" — and ``score()`` below deliberately
# routes exactly those rows to the judge (the ``is_true_abstention``
# short-circuit). So the harness sent the judge a partial answer and then
# instructed it to fail the row for containing the words it was sent for: any
# row whose text ends in the canned "I don't have enough information..." scored
# 0.0 no matter what it said BEFORE that sentence. Measured: ~5.3% of
# abstention-bearing rows are partial.
#
# The replacement keeps a TRUE abstention at correct=false — that is the other
# half of the same contract, and a judge that rewarded a refusal would be a
# worse instrument than one that punishes a partial answer. The distinction the
# judge is asked to draw is exactly ``is_true_abstention``'s, expressed in words
# it can apply to TEXT: does the answer state a substantive fact of its own, or
# only that it lacks one?
#
# ⚠️ This CHANGES the measuring instrument. `judge_correct` scored before this
# commit and after it are two different measurements on the partial-bearing
# subset; see the comparability note in eval/metrics.py.
_CORRECTNESS_SYSTEM = (
    "You grade a generated ANSWER against a GOLD answer for a factual question. "
    "The ANSWER may be verbose or phrased differently; it is CORRECT if it clearly "
    "conveys the gold fact (allowing paraphrase, extra correct detail, or a reasoned "
    'conclusion). Output ONLY JSON: {"correct": true|false, "reason": "<one sentence>"}. '
    "An ANSWER that ONLY declares it lacks the information, and states no substantive "
    "fact of its own, is correct=false. "
    "An ANSWER that answers part of the question and declares insufficiency only on the "
    "rest is NOT an abstention: grade the substantive content it DOES state, exactly as "
    "you would grade any other answer — correct=true if that content clearly conveys the "
    "gold fact, correct=false if it does not. "
    "The declaration of insufficiency is never itself evidence either way."
)


def _correctness_user(question: str, gold_answers: list[str], answer: str) -> str:
    golds = " | ".join(gold_answers) if gold_answers else "(none)"
    return f"QUESTION:\n{question}\n\nGOLD ANSWER(S):\n{golds}\n\nANSWER:\n{answer}\n"


async def score(
    example: Example,
    answer: str,
    *,
    judge: Judge | None,
    abstained: bool,
    abstain_reason: str | None = None,
) -> dict[str, Any]:
    """EM/F1/cover-EM + optional LLM correctness for one multi-hop answer."""
    gold = example.gold_answer or ""
    row: dict[str, Any] = {
        "em": metrics.exact_match(answer, gold),
        "cover_em": metrics.cover_em(answer, gold),
    }
    # Best over aliases (2Wiki/Hotpot usually single, but be safe).
    for alias in example.all_gold_answers()[1:]:
        row["em"] = max(row["em"], metrics.exact_match(answer, alias))
        row["cover_em"] = max(row["cover_em"], metrics.cover_em(answer, alias))

    # f1 is maxed over the SAME alias list, but its ceiling and attainment must
    # come from the ONE alias that produced the winning f1. Maxing them
    # independently (as em/cover_em are) would divide an f1 measured against a
    # 2-token gold by a ceiling measured against a 5-token alias, and the ratio
    # would mean nothing. Candidate order and tie-breaking (max keeps the first
    # maximal element, exactly as the running `max(...)` above does) are
    # identical to the old loop, so `f1` itself is bit-for-bit unchanged.
    f1_aliases = [gold, *example.all_gold_answers()[1:]]
    best_f1, best_ceiling, best_alias = max(
        ((metrics.token_f1(answer, a), metrics.token_f1_ceiling(answer, a), a) for a in f1_aliases),
        key=lambda scored: scored[0],
    )
    row["f1"] = best_f1
    row["f1_ceiling"] = best_ceiling
    row["f1_attainment"] = metrics.token_f1_attainment(answer, best_alias)

    if judge is not None:
        # Only a TRUE abstention short-circuits the judge. Forcing
        # judge_correct=0.0 on a `partial_context` row while em/f1/cover_em are
        # still computed from the real answer text produces a self-contradictory
        # row (f1>0 next to "not correct") -- a silently wrong metric.
        if metrics.is_true_abstention(abstained, abstain_reason):
            row["judge_correct"] = 0.0  # honest abstention is not "correct"
        else:
            # `require=("correct",)` is the contract, not a hint. A judge that
            # replies `{"reason": "..."}` parses fine, so before this the guard
            # below dropped the row in silence with nothing counted anywhere —
            # and a dropped row is a row off `n_scored`, which `make
            # bench-promote` requires at >= 50 for EVERY ladder rung. Declaring
            # the key makes that failure countable and therefore retryable.
            verdict = await judge.judge_json(
                _CORRECTNESS_SYSTEM,
                _correctness_user(example.question, example.all_gold_answers(), answer),
                require=("correct",),
            )
            # Belt and braces: `require` is enforced inside the judge, but this
            # guard is what keeps a keyless verdict from ever reaching the row,
            # and it must survive anyone setting KAG_EVAL_JUDGE_RETRIES=0 or
            # swapping in a stub judge that ignores `require`.
            if verdict is not None and "correct" in verdict:
                row["judge_correct"] = 1.0 if verdict.get("correct") else 0.0
                row["judge_reason"] = str(verdict.get("reason", ""))[:300]
    return row
