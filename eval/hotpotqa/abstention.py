"""Abstention detection for strata answers.

Strata's /api/ask has no abstain flag; its system prompt says "If the sources do not contain the
answer, say that you do not know." KAG's copied ``metrics.py`` carries no phrase logic either -
only the THREE-state contract (``is_true_abstention`` / ``PARTIAL_ABSTAIN_REASON``): a true
abstention carries no answer, a ``partial_context`` answer states some facts and declares
insufficiency on the rest and MUST be scored on its content.

The rule here maps strata text onto that contract:

* empty / whitespace answer                        -> abstained, reason None (true abstention)
* an "I don't know" phrase AND zero citations      -> abstained, reason None (true abstention)
* an "I don't know" phrase AND >= 1 citation       -> abstained, reason ``partial_context``
* otherwise                                        -> not abstained

Citations are the discriminator because strata's prompt makes the model cite every statement;
an answer that cites nothing stated nothing from the sources. ``is_true_abstention`` then decides
what the scorer does: only a true abstention short-circuits the judge to 0.0, exactly as KAG did.
"""

from __future__ import annotations

import re

import metrics

_IDK = re.compile(
    r"""
    \b(?:do|does|did)\s*n[o']?t\s+(?:know|contain|mention|include|provide|say|specify|state|
         indicate|give|list|identify|name|have)\b
    | \b(?:cannot|can\s*not|can't|unable\s+to)\s+(?:determine|answer|find|tell|identify|
         confirm|say|know|be\s+determined)\b
    | \b(?:no|not\s+enough|insufficient|not\s+sufficient)\s+(?:information|evidence|details?|
         mention|data)\b
    | \bnot\s+(?:mentioned|provided|specified|stated|included|available)\s+in\s+the\s+sources?\b
    | \bI\s+don't\s+know\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def declares_insufficiency(answer: str) -> bool:
    return bool(_IDK.search(answer or ""))


def classify(answer: str, n_citations: int) -> tuple[bool, str | None]:
    """``(abstained, abstain_reason)`` in the vocabulary of ``metrics.is_true_abstention``."""
    if not (answer or "").strip():
        return True, None
    if declares_insufficiency(answer):
        if n_citations > 0:
            return True, metrics.PARTIAL_ABSTAIN_REASON
        return True, None
    return False, None


def is_true_abstention(answer: str, n_citations: int) -> bool:
    return metrics.is_true_abstention(*classify(answer, n_citations))


def persisted(answer: str, n_citations: int) -> tuple[bool, str | None]:
    """``abstained`` / ``abstain_reason`` as persisted in ``results.jsonl``.

    ``abstained`` is ``metrics.is_true_abstention`` of the classification, so a
    ``partial_context`` row is written ``abstained=False`` with its reason kept. That is what the
    README and KAG's summary mean by "abstained", and ``metrics.row_is_true_abstention`` reads the
    same verdict back (``is_true_abstention(False, partial_context)`` is False). Passing these two
    fields to ``qa.score`` is equivalent to passing ``classify``'s output.
    """
    abstained, reason = classify(answer, n_citations)
    return metrics.is_true_abstention(abstained, reason), reason
