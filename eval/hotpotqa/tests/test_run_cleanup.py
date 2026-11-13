"""run.py against a mocked strata (respx): the store is always deleted unless --keep-store."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

import run
from examples import DATA_FILE

BASE = "http://strata.test"
ROOT = "00000000-0000-7000-8000-000000000001"
STORE = "00000000-0000-7000-8000-0000000000aa"


@pytest.fixture
def two_examples(tmp_path: Path) -> Path:
    lines = DATA_FILE.read_text().splitlines()[:2]
    p = tmp_path / "n2.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _mock_tree(mock: respx.MockRouter) -> respx.Route:
    mock.get("/health").respond(200, json={"status": "ok"})
    mock.get("/api/stores").respond(
        200,
        json={"stores": [{"id": ROOT, "parent_id": None, "kind": "router", "name": "library"}]},
    )
    mock.post("/api/stores").respond(201, json={"id": STORE, "parent_id": ROOT, "kind": "leaf"})
    return mock.delete(f"/api/stores/{STORE}").respond(204)


def _ingest_ok(mock: respx.MockRouter) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={"status": "created", "id": f"doc-{counter['n']}", "source": body["source"]},
        )

    mock.post(f"/api/stores/{STORE}/documents").mock(side_effect=handler)


def _args(data: Path, out: Path, *extra: str) -> list[str]:
    return ["--base-url", BASE, "--no-judge", "--data", str(data), "--out", str(out), *extra]


def test_store_deleted_when_ingest_fails(two_examples: Path, tmp_path: Path) -> None:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        delete = _mock_tree(mock)
        mock.post(f"/api/stores/{STORE}/documents").respond(
            500, json={"error": {"code": "internal", "message": "boom"}}
        )
        with pytest.raises(Exception, match="HTTP 500"):
            run.main(_args(two_examples, tmp_path / "out"))
        assert delete.call_count == 1


def test_store_deleted_on_keyboard_interrupt(two_examples: Path, tmp_path: Path) -> None:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        delete = _mock_tree(mock)
        _ingest_ok(mock)

        def interrupt(request: httpx.Request) -> httpx.Response:
            raise KeyboardInterrupt

        mock.post("/api/ask").mock(side_effect=interrupt)
        assert run.main(_args(two_examples, tmp_path / "out")) == 130
        assert delete.call_count == 1


def test_keep_store_skips_delete(two_examples: Path, tmp_path: Path) -> None:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        delete = _mock_tree(mock)
        mock.post(f"/api/stores/{STORE}/documents").respond(500, json={})
        with pytest.raises(Exception, match="HTTP 500"):
            run.main(_args(two_examples, tmp_path / "out", "--keep-store"))
        assert delete.call_count == 0


def test_full_run_writes_text_free_results_and_deletes_store(
    two_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        delete = _mock_tree(mock)
        _ingest_ok(mock)
        asks: list[dict[str, object]] = []

        def ask(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            asks.append(body)
            if len(asks) == 2:
                return httpx.Response(502, json={"error": {"code": "upstream_unavailable"}})
            return httpx.Response(
                200,
                json={
                    "answer": "It is doc one [1].",
                    "citations": ["c1"],
                    "results": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc-1",
                            "store_path": ["library", "hotpotqa-n50"],
                            "text": "T",
                        },
                        {
                            "chunk_id": "c2",
                            "document_id": "doc-2",
                            "store_path": ["library", "hotpotqa-n50"],
                            "text": "T",
                        },
                    ],
                    "timings_ms": {
                        "embed": 1,
                        "route": 0,
                        "search": 2,
                        "rerank": 3,
                        "total": 7,
                        "generate": 40,
                    },
                },
            )

        mock.post("/api/ask").mock(side_effect=ask)
        assert run.main(_args(two_examples, out)) == 0
        assert delete.call_count == 1

    # the request is FLAT and leaf scope pins root_id to the benchmark store
    assert asks[0]["root_id"] == STORE and asks[0]["top_k"] == 10
    assert asks[0]["route_mode"] == "hybrid" and asks[0]["max_tokens"] == 800
    assert isinstance(asks[0]["query"], str)

    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    assert len(errs) == 1 and "HTTP 502" in errs[0]["error"]
    r = ok[0]
    assert r["latency_ms_query"] == 7.0 and r["latency_ms_generate"] == 40.0
    assert r["n_citations"] == 1 and r["abstained"] is False
    assert r["leaves_used"] is None and r["laya_consulted"] is None  # not exposed by /api/ask
    assert 0.0 <= r["recall_at_5"] <= 1.0
    text = (out / "results.jsonl").read_text()
    assert "It is doc one" not in text and '"question":' not in text
    answers = [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines()]
    assert any(a.get("answer") == "It is doc one [1]." for a in answers)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_errors"] == 1 and meta["ingest"]["created"] == meta["ingest"]["n_passages"]


def test_tree_scope_sends_null_root(two_examples: Path, tmp_path: Path) -> None:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        _mock_tree(mock)
        _ingest_ok(mock)
        route = mock.post("/api/ask").respond(
            200, json={"answer": "", "citations": [], "results": [], "timings_ms": {}}
        )
        assert run.main(_args(two_examples, tmp_path / "o", "--scope", "tree", "--limit", "1")) == 0
        body = json.loads(route.calls[0].request.content)
        assert body["root_id"] is None
    row = json.loads((tmp_path / "o" / "results.jsonl").read_text())
    assert row["abstained"] is True and row["recall_at_5"] == 0.0


def test_existing_store_without_reuse_is_refused_and_not_deleted(
    two_examples: Path, tmp_path: Path
) -> None:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        mock.get("/health").respond(200, json={})
        mock.get("/api/stores").respond(
            200,
            json={
                "stores": [
                    {"id": ROOT, "parent_id": None, "kind": "router", "name": "library"},
                    {"id": STORE, "parent_id": ROOT, "kind": "leaf", "name": "hotpotqa-n50"},
                ]
            },
        )
        delete = mock.delete(f"/api/stores/{STORE}").respond(204)
        with pytest.raises(RuntimeError, match="already exists"):
            run.main(_args(two_examples, tmp_path / "out"))
        assert delete.call_count == 0
