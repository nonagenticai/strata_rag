"""document_id -> passage id mapping and the recall@k it feeds."""

from __future__ import annotations

import metrics
from strata_client import DocIndex, predicted_ranking


def _res(*docs: str) -> list[dict[str, str]]:
    return [{"document_id": d, "chunk_id": f"c{i}"} for i, d in enumerate(docs)]


def _index(pairs: list[tuple[str, str]]) -> DocIndex:
    idx = DocIndex()
    seen: set[str] = set()
    for doc, src in pairs:
        idx.add(doc, src, "existing" if doc in seen else "created")
        seen.add(doc)
    return idx


def test_rank_order_and_chunk_dedup() -> None:
    idx = _index([("d1", "q:A:0"), ("d2", "q:A:1"), ("d3", "q:A:2")])
    r = predicted_ranking(_res("d2", "d2", "d1", "d2", "d3"), idx, ["q:A:1", "q:A:2"])
    assert r.passage_ids == ["q:A:1", "q:A:0", "q:A:2"]
    assert metrics.recall_at_k(r.passage_ids, ["q:A:1", "q:A:2"], 5) == 1.0
    assert metrics.recall_at_k(r.passage_ids, ["q:A:1", "q:A:2"], 2) == 0.5
    assert r.collisions_top5 == 0 and r.unmapped_docs == 0


def test_dedup_collision_counts_for_the_gold_source() -> None:
    # Example B's gold paragraph is byte-identical to example A's passage 4: strata stored ONE
    # document (first ingested as A's) and answered "existing" for B's POST.
    idx = _index([("dX", "q:A:4"), ("dX", "q:B:1"), ("dY", "q:B:0")])
    assert idx.sources["dX"] == ["q:A:4", "q:B:1"]
    assert idx.created == 2 and idx.existing == 1
    assert idx.collision_docs() == 1 and idx.collided_passages() == 2
    gold_b = ["q:B:1", "q:B:0"]
    r = predicted_ranking(_res("dX", "dY"), idx, gold_b)
    assert r.passage_ids == ["q:B:1", "q:B:0"]
    assert metrics.recall_at_k(r.passage_ids, gold_b, 5) == 1.0
    assert r.collisions_top5 == 1
    assert r.collision_gold_credits_top5 == 1  # credit strict first-source identity would deny
    # For example A the same document is A's passage via its FIRST source: not a "collision credit".
    ra = predicted_ranking(_res("dX"), idx, ["q:A:4"])
    assert ra.passage_ids == ["q:A:4"]
    assert ra.collisions_top5 == 1 and ra.collision_gold_credits_top5 == 0


def test_collision_doc_without_gold_falls_back_to_first_source() -> None:
    idx = _index([("dX", "q:A:4"), ("dX", "q:B:1")])
    r = predicted_ranking(_res("dX"), idx, ["q:C:0"])
    assert r.passage_ids == ["q:A:4"]
    assert metrics.recall_at_k(r.passage_ids, ["q:C:0"], 5) == 0.0


def test_one_document_fills_one_rank_only() -> None:
    # A doc standing for two gold ids of the SAME question still occupies one rank and is
    # credited once, so recall cannot exceed what distinct documents earned.
    idx = _index([("dX", "q:A:0"), ("dX", "q:A:1")])
    r = predicted_ranking(_res("dX"), idx, ["q:A:0", "q:A:1"])
    assert r.passage_ids == ["q:A:0"]
    assert metrics.recall_at_k(r.passage_ids, ["q:A:0", "q:A:1"], 5) == 0.5


def test_unmapped_document_keeps_its_rank() -> None:
    idx = _index([("d1", "q:A:0"), ("d2", "q:A:1")])
    r = predicted_ranking(_res("zz1", "zz2", "zz3", "zz4", "zz5", "d1"), idx, ["q:A:0"])
    assert r.unmapped_docs == 5
    assert r.passage_ids[:5] == [f"doc:zz{i}" for i in range(1, 6)]
    assert metrics.recall_at_k(r.passage_ids, ["q:A:0"], 5) == 0.0
    assert metrics.recall_at_k(r.passage_ids, ["q:A:0"], 10) == 1.0
