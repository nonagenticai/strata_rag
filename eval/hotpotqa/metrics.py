# copied from kag@d653afba14d3f3c82ef84dea1a88f226d20e5726
# eval/metrics.py, unchanged except imports
"""Pure metric functions for the eval harness.

Recall@k, MRR, nDCG@k operate on a predicted ranking + a set of relevant
chunk_ids. Cohen's kappa for judge sanity (rerunning the judge on the
same set should produce highly correlated scores; low kappa means the
judge prompt is unstable).

The QA block (``normalize_answer``, ``exact_match``, ``token_f1``,
``cover_em``) is the SQuAD/HotpotQA-style short-answer scoring used by the
multi-hop benchmark (HiRAG paper reports EM + F1). ``token_f1_ceiling`` /
``token_f1_attainment`` say how much of that F1 a prose answer of this length
COULD have scored — without them a raw F1 is compared to an anchor the answer
format cannot reach. ``rmse`` correlates our LLM-judge TRACe scores against
RAGBench's shipped gold labels.

⚠️ Every PREDICTION-side token metric normalises through
``normalize_prediction``, which strips the inline citation markers KAG's
synthesis prompt mandates BEFORE the SQuAD normalisation eats their brackets.
That is a comparability break against runs scored before 2026-09-16 — see the
block above ``strip_citation_markers``.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from math import log2, sqrt

# ---------------------------------------------------------------------------
# Abstention semantics (shared by the scorers, the report and the summary).
# ---------------------------------------------------------------------------
# `abstained` is a bool over a THREE-state world (see models/ask.py
# AskDiagnostics). `no_matching_content` and `insufficient_context` carry no
# answer; `partial_context` DOES -- the model answered part of the question
# with citations and declared insufficiency only on the rest.
#
# Every consumer that suppresses content scoring must therefore branch on the
# REASON, not on `abstained` alone. Scoring a partial answer as CRAG "missing",
# judge_correct=0.0, or counting it in abstain_rate is a SILENTLY WRONG METRIC:
# the row carries a real, citable answer that must be judged on its content
# like any other, and skipping the hallucination / missing-citation checks
# leaves a real defect undetected. Measured on eval/_runs/: 74 of 1,387
# abstention-bearing answers are partial (~2.6% of all 2,709 rows), so treating
# them as abstentions moves aggregate scores across a version boundary.
PARTIAL_ABSTAIN_REASON = "partial_context"


def is_true_abstention(abstained: object, abstain_reason: object = None) -> bool:
    """True when this row abstained AND carries no answer to score.

    Backwards compatible with rows persisted before `abstain_reason` existed:
    a missing/None reason with `abstained=True` is a true abstention, exactly
    as those runs were scored, so historical runs stay comparable.
    """
    return bool(abstained) and abstain_reason != PARTIAL_ABSTAIN_REASON


def row_is_true_abstention(row: Mapping[str, object]) -> bool:
    """`is_true_abstention` over a persisted result row (see `_load`)."""
    return is_true_abstention(row.get("abstained"), row.get("abstain_reason"))


def recall_at_k(predicted: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    hits = sum(1 for cid in list(predicted)[:k] if cid in rel)
    return hits / len(rel)


def mrr(predicted: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, cid in enumerate(predicted, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(predicted: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    dcg = 0.0
    for i, cid in enumerate(list(predicted)[:k], start=1):
        if cid in rel:
            dcg += 1.0 / log2(i + 1)
    ideal = sum(1.0 / log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for ordinal scores (1-5). Returns 0 on degenerate input."""
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    # Marginal expected agreement.
    a_dist: dict[int, float] = {}
    b_dist: dict[int, float] = {}
    for x in a:
        a_dist[x] = a_dist.get(x, 0) + 1 / n
    for y in b:
        b_dist[y] = b_dist.get(y, 0) + 1 / n
    expected = sum(a_dist.get(k, 0) * b_dist.get(k, 0) for k in set(a) | set(b))
    if expected == 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - expected) / (1 - expected)


# ---------------------------------------------------------------------------
# Short-answer QA metrics (multi-hop: HotpotQA / 2WikiMultiHopQA).
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Inline citation markers are INSTRUMENT ARTEFACTS, not answer content.
# ---------------------------------------------------------------------------
# `SYNTH_PROMPT_TEMPLATE` (src/services/ask_pipeline.py, rule 1) MANDATES them:
# "Every factual claim must be followed by a citation marker like [1] or [2,5]
# referring to the chunk numbers, or [G1], [G2] for graph facts." So every KAG
# answer carries markers the gold answer can never carry, and `normalize_answer`
# below strips `string.punctuation` -- which turns `[3]` into the bare token `3`
# rather than removing it. That moved every token metric in BOTH directions:
#
#   cover_em("The city is Shanghai [1], China [2].", "Shanghai, China") -> 0.0
#       FALSE NEGATIVE: the gold bigram is split by an interposed `1`.
#   cover_em("He was born on May 20 [4], 1999.", "May 20, 1999")        -> 0.0
#       FALSE NEGATIVE: same, inside a date.
#   cover_em("The answer appears in chunk [3] ...", "3")                -> 1.0
#       FALSE POSITIVE: a marker IS the gold answer, so an answer that never
#       states the fact scores a perfect hit.
#   token_f1: every marker is an extra prediction token, so the PRECISION
#       denominator is inflated by the model obeying its instructions.
#
# The two alternatives below are EXACTLY the two forms
# `ask_pipeline._parse_citations` (ask_pipeline.py:1437,1458) recognises:
# `\[([0-9,\s]+)\]` for chunk markers (`[1]`, `[2,5]`, `[2, 5]`) and
# `\[G([0-9]+)\]` for graph-fact markers (`[G1]`). Deliberately no wider:
# bracketed text containing any LETTER other than a leading `G`+digits -- `[sic]`,
# `[1990s]`, `[Editor's note]` -- is answer content and is left alone. A
# combination the pipeline itself does not parse (`[1, G2]`) is likewise left
# alone, so this function and the shipped parser can never disagree about what a
# citation is.
#
# ⚠️ Residual, stated rather than hidden: a legitimate bracketed bare number in
# an answer (`[2012]`) is stripped. That is the same string the shipped
# `_parse_citations` would also read as a citation attempt, so the two layers
# stay consistent; no gold answer in the multi-hop benchmarks is bracketed.
_CITATION_MARKER = re.compile(r"\[[0-9,\s]+\]|\[G[0-9]+\]")

# ⛔ COMPARABILITY BREAK -- READ BEFORE COMPARING ACROSS THIS COMMIT.
# Every token metric computed from a PREDICTION changes value here: `em`, `f1`,
# `f1_ceiling`, `f1_attainment` and `cover_em`. BOTH promoted baselines --
# `eval/_runs/baselines/n50-20260830.{summary.json,report.md}` and
# `eval/_runs/baselines/n50-20260902.{summary.json,report.md}` (the only two
# files in that directory) -- plus the committed regression fixture
# `tests/eval/fixtures/n50-20260830-scores.jsonl` were scored with the OLD
# behaviour, and those per-row values are FROZEN in their `results.jsonl`:
# `eval/summary.py` and `eval/subset.py` re-AGGREGATE stored per-row numbers and
# never re-SCORE, so a
# `report.py --baseline <old results.jsonl>` run silently averages old-definition
# baseline rows against new-definition current rows. Neither `eval/promote.py`
# nor `eval/subset.py` carries a metric-version field to catch that (promote
# reads only `gate` / ladder `n` / `h1.paired_n` / `baseline_paired.error`;
# subset's `gold_parity` compares ingested CORPUS parity, not metric parity).
# A baseline scored before this commit must be RE-SCORED, not re-aggregated,
# before its `em`/`f1`/`cover_em` are compared to a run scored after it.
# `judge_correct`, `recall_at_5`, `ndcg_at_10` and the TRACe/CRAG scores are
# untouched and stay comparable.


def strip_citation_markers(s: str) -> str:
    """Remove the inline citation markers KAG's synthesis prompt mandates.

    Applied to the PREDICTION only, never to the gold answer (a gold answer has
    no markers, and stripping a bracketed gold would be a different change).
    Markers become a single SPACE, not the empty string: `Shanghai[1]China`
    must not normalise to the single token `shanghaichina`.
    """
    return _CITATION_MARKER.sub(" ", s or "")


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation + articles,
    collapse whitespace. The canonical form for EM/F1 on short answers.

    ⛔ Does NOT strip citation markers: it is applied to GOLD answers too, and
    gold carries none. Prediction-side normalisation goes through
    :func:`normalize_prediction`.
    """
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalize_prediction(s: str) -> str:
    """:func:`normalize_answer`, with citation markers removed FIRST.

    Order matters: `normalize_answer` deletes `[` and `]` as punctuation, so
    after it runs a marker is indistinguishable from a bare number the model
    actually wrote. Every prediction-side token metric in this module uses this.
    """
    return normalize_answer(strip_citation_markers(s))


def exact_match(prediction: str, gold: str) -> float:
    """1.0 iff the normalized prediction equals the normalized gold answer."""
    return 1.0 if normalize_prediction(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 on normalized answers (SQuAD/HotpotQA definition).

    Multiplicity-aware (matches min count per shared token), so a prediction
    that repeats a gold word does not over-credit. Returns 0.0 when either
    side is empty unless both are empty (then 1.0), mirroring the SQuAD eval.
    """
    pred_toks = normalize_prediction(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        # If both are empty they trivially match; otherwise no overlap.
        return 1.0 if pred_toks == gold_toks else 0.0
    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    pred_counts: dict[str, int] = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        if t in gold_counts:
            common[t] = min(c, gold_counts[t])
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1_ceiling(prediction: str, gold: str) -> float:
    """The MAXIMUM token_f1 achievable for this prediction length.

    Assumes every gold token is present in the prediction (perfect recall).
    A long generative answer against a 1-3 token extractive gold is bounded
    far below 1.0 no matter how correct it is, so a raw f1 cannot be compared
    to an extractive-QA anchor without this denominator.

    Closed form: with L = |pred_toks| and g = |gold_toks|, perfect recall means
    num_same = min(L, g), so the bound is ``2*min(g, L) / (L + g)``. At g=2,
    L=43 (the measured full-mode median) that is 0.089 — a PERFECT answer
    cannot beat 0.089, while ``baselines.py`` anchors `strong` at 0.55.
    """
    pred_toks = normalize_prediction(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        # Mirror token_f1's empty-input contract exactly: both empty trivially
        # match (and a trivially-matched pair has a ceiling of 1.0); otherwise
        # nothing is attainable.
        return 1.0 if pred_toks == gold_toks else 0.0
    num_same = min(len(pred_toks), len(gold_toks))
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1_attainment(prediction: str, gold: str) -> float | None:
    """`token_f1` as a FRACTION of what this prediction length could attain.

    This is the number that answers "is the answer wrong, or is the metric
    saturated?". A raw f1 of 0.084 against a ceiling of 0.089 is 94% attainment
    — a near-perfect answer scored by a metric that cannot express it.

    Returns None (not 0.0, not 1.0) when the ceiling is 0.0 — an empty gold
    against a non-empty prediction. Nothing is attainable there, so there is no
    fraction to report and the row must be excluded from the mean rather than
    given a fabricated value.
    """
    ceiling = token_f1_ceiling(prediction, gold)
    if ceiling == 0.0:
        return None
    # token_f1's num_same = sum(min(pred_count, gold_count)) is bounded above by
    # both len(pred_toks) and len(gold_toks), i.e. by exactly the min(L, g) the
    # ceiling assumes. The ratio therefore cannot exceed 1.0 mathematically;
    # only float division noise can push it a few ulps over, so clamp (an assert
    # would turn a rounding artefact into a crashed benchmark run).
    return min(1.0, token_f1(prediction, gold) / ceiling)


def cover_em(prediction: str, gold: str) -> float:
    """Lenient "answer recall": 1.0 iff the normalized gold answer appears as a
    contiguous token subsequence of the normalized prediction.

    KAG synthesizes prose, not a short span, so strict EM under-credits a
    correct answer buried in a sentence. cover_em is the standard companion
    metric (a.k.a. "answer-in-generation") for generative RAG on extractive QA.

    The prediction goes through :func:`normalize_prediction`: an interposed
    citation marker breaks the CONTIGUITY this metric is defined on, so a
    marker-bearing answer scored 0.0 while stating the gold answer verbatim.
    """
    n_pred = normalize_prediction(prediction)
    n_gold = normalize_answer(gold)
    if not n_gold:
        return 0.0
    pred_toks = n_pred.split()
    gold_toks = n_gold.split()
    if not gold_toks:
        return 0.0
    for i in range(len(pred_toks) - len(gold_toks) + 1):
        if pred_toks[i : i + len(gold_toks)] == gold_toks:
            return 1.0
    return 0.0


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    """Root-mean-square error between two equal-length score sequences.

    Used to report how far our LLM-judge TRACe scores sit from RAGBench's
    shipped gold labels (lower = the judge tracks the gold better). Returns
    0.0 on degenerate/mismatched input so a bad row never aborts a run.
    """
    if len(a) != len(b) or not a:
        return 0.0
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a))


__all__ = [
    "cohens_kappa",
    "cover_em",
    "exact_match",
    "mrr",
    "ndcg_at_k",
    "normalize_answer",
    "normalize_prediction",
    "recall_at_k",
    "rmse",
    "strip_citation_markers",
    "token_f1",
    "token_f1_attainment",
    "token_f1_ceiling",
]
