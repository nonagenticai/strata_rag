# copied from kag@d653afba14d3f3c82ef84dea1a88f226d20e5726 eval/benchmarks/common.py
# (Example/Passage/ext_id, trimmed) and eval/benchmarks/multihop.py (row_to_example /
# _extract_context path), code unchanged except imports; long docstrings shortened (see the
# originals for their rationale). One deliberate difference:
# ext_id drops KAG's "kageval:" tenant prefix, giving "hotpotqa:<example_id>:<passage_idx>".
"""HotpotQA rows -> ``Example`` objects with deterministic passage ids.

One strata document per passage; the document ``source`` is :func:`ext_id`. Gold passages are
those whose title appears in ``supporting_facts``; ``passage_idx`` is the index in
``Example.passages`` AFTER empty paragraphs are dropped, exactly as KAG numbered them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _slug(s: str) -> str:
    """Sanitize an id fragment to the ``[A-Za-z0-9_.-]`` charset."""
    return _SAFE.sub("-", str(s)).strip("-") or "x"


def ext_id(benchmark: str, example_id: str, passage_idx: int) -> str:
    """Deterministic id for one ingested passage: ``<benchmark>:<example_id>:<idx>``."""
    return f"{_slug(benchmark)}:{_slug(example_id)}:{passage_idx}"


@dataclass
class Passage:
    """One retrievable unit of an example's corpus (becomes one strata document)."""

    text: str
    title: str | None = None
    is_gold: bool = False


@dataclass
class Example:
    """A single benchmark question with its corpus + gold signal."""

    id: str
    benchmark: str
    question: str
    passages: list[Passage] = field(default_factory=list)
    gold_answer: str | None = None
    gold_answers: list[str] = field(default_factory=list)
    domain: str | None = None
    question_type: str | None = None
    meta: dict = field(default_factory=dict)

    def gold_external_ids(self) -> list[str]:
        """ids of the gold-supporting passages (relevant set for IR metrics)."""
        return [
            ext_id(self.benchmark, self.id, i) for i, p in enumerate(self.passages) if p.is_gold
        ]

    def passage_ext_ids(self) -> list[str]:
        """ids of ALL this example's passages, in order."""
        return [ext_id(self.benchmark, self.id, i) for i in range(len(self.passages))]

    def all_gold_answers(self) -> list[str]:
        """Primary + alias answers, de-duplicated, primary first."""
        out: list[str] = []
        for a in [self.gold_answer, *self.gold_answers]:
            if a and a not in out:
                out.append(a)
        return out


# --- eval/benchmarks/multihop.py (code unchanged) -------------------------------------------------


def _dejson(x: object) -> object:
    """Decode a JSON-encoded string field if that's what it is."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def _decode_field(value: object, name: str) -> object:
    """Decode a whole ``context`` / ``supporting_facts`` field that arrived as a JSON string."""
    original = value
    if isinstance(value, bytes | bytearray):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return value
    if not value.strip():
        return None
    decoded = _dejson(value)
    if isinstance(decoded, dict | list | tuple):
        return decoded
    raise ValueError(
        f"multihop row: {name!r} is a {type(original).__name__} that does not decode to the "
        f"dict form or the list form (got {type(decoded).__name__}: {value[:120]!r}). "
        "Refusing to iterate it as characters, which would yield zero passages and "
        "silently drop the example."
    )


def _sentences(value: object) -> list[str]:
    """One document's sentence list, decoding a JSON-encoded leaf."""
    decoded = _dejson(value)
    if isinstance(decoded, list | tuple):
        return [str(s) for s in decoded]
    return [str(decoded)] if decoded else []


def _context_to_passages(
    titles: list[str], sentence_lists: list[list[str]], gold_titles: set[str]
) -> list[Passage]:
    passages: list[Passage] = []
    for title, sents in zip(titles, sentence_lists, strict=False):
        text = " ".join(s.strip() for s in (sents or []) if s).strip()
        if not text:
            continue
        passages.append(Passage(text=text, title=title, is_gold=title in gold_titles))
    return passages


def _seq(value: object) -> Sequence[object]:
    """``value`` when it is a real sequence container, else empty."""
    return value if isinstance(value, list | tuple) else []


def _extract_context(
    context: object, supporting_facts: object
) -> tuple[list[str], list[list[str]], set[str]]:
    """Return (titles, sentence_lists, gold_titles) from either the dict form
    (parallel arrays) or the list form (list of [title, sentences] pairs)."""
    context = _decode_field(context, "context")
    supporting_facts = _decode_field(supporting_facts, "supporting_facts")

    titles: list[str] = []
    sentence_lists: list[list[str]] = []
    if isinstance(context, dict):
        titles = [str(t) for t in _seq(_decode_field(context.get("title"), "context['title']"))]
        raw_sents = _decode_field(
            context.get("sentences") or context.get("content"), "context['sentences']"
        )
        sentence_lists = [_sentences(doc) for doc in _seq(raw_sents)]
        sf = supporting_facts if isinstance(supporting_facts, dict) else {}
        gold_titles = {
            str(t) for t in _seq(_decode_field(sf.get("title"), "supporting_facts['title']"))
        }
        return titles, sentence_lists, gold_titles

    for item in _seq(context):
        if isinstance(item, list | tuple) and len(item) >= 2:
            titles.append(str(_dejson(item[0])))
            sentence_lists.append(_sentences(item[1]))
    gold_titles = {
        str(_dejson(sf[0])) for sf in _seq(supporting_facts) if isinstance(sf, list | tuple) and sf
    }
    return titles, sentence_lists, gold_titles


def row_to_example(row: dict, *, benchmark: str) -> Example:
    """Pure converter for a Hotpot row (dict or list context form)."""
    titles, sentence_lists, gold_titles = _extract_context(
        row.get("context"), row.get("supporting_facts")
    )
    passages = _context_to_passages(titles, sentence_lists, gold_titles)
    return Example(
        id=str(row.get("id") or row.get("_id") or ""),
        benchmark=benchmark,
        question=str(row.get("question") or ""),
        passages=passages,
        gold_answer=str(row.get("answer") or ""),
        domain=str(row.get("type") or "_all"),
        question_type=str(row.get("type") or ""),
        meta={"level": row.get("level")},
    )


# --- strata-specific ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "n50.jsonl"


def load_examples(path: Path = DATA_FILE) -> list[Example]:
    """The committed fixture as Examples, in dataset order."""
    out: list[Example] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(row_to_example(json.loads(line), benchmark="hotpotqa"))
    return out
