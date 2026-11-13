"""Async httpx client for the strata HTTP API, plus the passage-id mapping used for recall.

API facts this client relies on (read from src/api/handlers.rs, src/query.rs, src/ask.rs,
src/engine.rs, src/ingest.rs at 42002ec):

* ``POST /api/stores`` ``{parent_id, kind, name, description}`` -> 201 StoreView. Strata allows
  exactly ONE root store (``parent_id: null``); a second root is 422 "a root store already
  exists". So the benchmark leaf is created UNDER the existing root router when there is one,
  and only becomes the root on an empty server. Sibling names are unique (409 duplicate_name).
* ``GET /api/stores`` -> ``{"stores": [StoreView...]}``, flat, each with ``parent_id``/``children``.
* ``POST /api/stores/{id}/documents`` JSON ``{text, source, title}`` -> 201
  ``{"status":"created", id, store_id, source, title, token_count, chunk_count}``, or 200
  ``{"status":"existing", ...}`` when the normalised text's sha256 already exists IN THAT STORE.
  An "existing" reply carries the FIRST document's id and source, not the one we sent.
  Ingest is synchronous (embedded and committed before the reply).
* ``POST /api/ask`` is FLAT (the query fields are ``#[serde(flatten)]``): ``{query, top_k,
  route_mode, beam, min_score, root_id, max_tokens}`` -> ``{answer, citations[chunk uuid],
  results[{chunk_id, document_id, store_path, text, scores}], timings_ms{embed, route, search,
  rerank, total, generate}}`` (``timings_ms`` is flattened too). There is NO ``routing`` block
  in the ask reply, so leaves used / laya consulted are not observable through /api/ask.
* ``route_mode`` is one of ``embed | laya | hybrid``; ``top_k`` is 1..=50.
* ``DELETE /api/stores/{id}`` -> 204. Documents and chunks go with it (FK ``ON DELETE CASCADE``);
  a leaf has no children so ``?recursive`` is not needed. 404 when already gone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

ROUTE_MODES = ("embed", "laya", "hybrid")


class StrataApiError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, body: str) -> None:
        self.status = status
        self.code = ""
        try:
            err = httpx.Response(status, content=body.encode()).json().get("error") or {}
            self.code = str(err.get("code") or "")
            msg = str(err.get("message") or body)
        except (ValueError, AttributeError):
            msg = body
        super().__init__(f"{method} {path} -> HTTP {status} {self.code}: {msg[:300]}")


class StrataClient:
    def __init__(self, base_url: str, *, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> StrataClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _req(self, method: str, path: str, **kw: Any) -> httpx.Response:
        resp = await self._http.request(method, path, **kw)
        if resp.status_code >= 400:
            raise StrataApiError(method, path, resp.status_code, resp.text)
        return resp

    async def health(self) -> dict[str, Any]:
        return (await self._req("GET", "/health")).json()

    async def list_stores(self) -> list[dict[str, Any]]:
        return list((await self._req("GET", "/api/stores")).json().get("stores") or [])

    async def find_store(self, name: str) -> dict[str, Any] | None:
        """The store named ``name`` directly under the root (or the root itself), if any."""
        stores = await self.list_stores()
        root = next((s for s in stores if s.get("parent_id") is None), None)
        for s in stores:
            if s.get("name") != name:
                continue
            if s.get("parent_id") is None or (root and s.get("parent_id") == root["id"]):
                return s
        return None

    async def create_leaf(self, name: str, description: str) -> dict[str, Any]:
        """Create the benchmark leaf under the root router, or as the root on an empty server."""
        stores = await self.list_stores()
        root = next((s for s in stores if s.get("parent_id") is None), None)
        if root is not None and root.get("kind") != "router":
            raise RuntimeError(
                f"the strata root store {root.get('name')!r} is a leaf; a second top-level store "
                "cannot be created. Use an empty server or one whose root is a router."
            )
        body = {
            "parent_id": root["id"] if root else None,
            "kind": "leaf",
            "name": name,
            "description": description,
        }
        return (await self._req("POST", "/api/stores", json=body)).json()

    async def ingest(
        self, store_id: str, *, text: str, source: str, title: str | None
    ) -> dict[str, Any]:
        body = {"text": text, "source": source, "title": title}
        return (await self._req("POST", f"/api/stores/{store_id}/documents", json=body)).json()

    async def ask(
        self,
        question: str,
        *,
        top_k: int,
        route_mode: str,
        root_id: str | None,
        max_tokens: int,
        beam: int | None = None,
        min_score: float | None = None,
        expand: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": question,
            "top_k": top_k,
            "route_mode": route_mode,
            "root_id": root_id,
            "max_tokens": max_tokens,
        }
        if beam is not None:
            body["beam"] = beam
        if min_score is not None:
            body["min_score"] = min_score
        if expand:
            body["expand"] = True
        return (await self._req("POST", "/api/ask", json=body)).json()

    async def delete_store(self, store_id: str) -> bool:
        """True if deleted, False if it was already gone."""
        try:
            await self._req("DELETE", f"/api/stores/{store_id}")
        except StrataApiError as e:
            if e.status == 404:
                return False
            raise
        return True


def delete_store_sync(base_url: str, store_id: str, *, timeout_s: float = 60.0) -> bool:
    """Blocking delete for ``finally`` / KeyboardInterrupt paths, where the event loop is gone."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s) as http:
        resp = http.delete(f"/api/stores/{store_id}")
    if resp.status_code == 404:
        return False
    if resp.status_code >= 400:
        raise StrataApiError("DELETE", f"/api/stores/{store_id}", resp.status_code, resp.text)
    return True


# --- document_id -> passage id mapping ----------------------------------------------------------


@dataclass
class DocIndex:
    """strata ``document_id`` -> every passage id (``source``) that was ingested AS it.

    Strata dedups by sha256 of the normalised text within a store, so two HotpotQA passages with
    identical text (the same Wikipedia paragraph shown to two questions) become ONE document. The
    second POST answers ``existing`` with the first document's id; both sources are recorded
    here, so that document counts as whichever of its passages is gold for the question asked.
    """

    sources: dict[str, list[str]] = field(default_factory=dict)
    created: int = 0
    existing: int = 0

    def add(self, document_id: str, source: str, status: str) -> None:
        srcs = self.sources.setdefault(document_id, [])
        if source not in srcs:
            srcs.append(source)
        if status == "created":
            self.created += 1
        else:
            self.existing += 1

    def collision_docs(self) -> int:
        """Documents that stand for more than one passage id."""
        return sum(1 for s in self.sources.values() if len(s) > 1)

    def collided_passages(self) -> int:
        """Passage ids that share their document with at least one other passage id."""
        return sum(len(s) for s in self.sources.values() if len(s) > 1)


@dataclass
class Ranking:
    """Predicted passage ids, one per distinct retrieved document, in rank order."""

    passage_ids: list[str]
    collisions_top5: int  # top-5 documents standing for >1 passage id
    collision_gold_credits_top5: int  # ...of which the gold credit came via a non-first source
    unmapped_docs: int  # retrieved documents not in the index (another store, or a stale run)


def predicted_ranking(
    results: Sequence[Mapping[str, Any]], index: DocIndex, gold: Iterable[str]
) -> Ranking:
    """Map ask ``results`` (chunks, rank order) to passage ids for the IR metrics.

    Dedup by ``document_id`` keeping first-seen order (a long passage can be several chunks).
    A document mapping to several passage ids is represented by a gold one when it has one not
    already credited, else by its first id; each document fills exactly ONE rank, so recall@k
    is still over the top-k distinct documents. Unmapped documents keep their rank as a
    non-gold placeholder (``doc:<uuid>``) rather than being dropped, which would promote lower
    ranks into the top k.
    """
    gold_set = set(gold)
    credited: set[str] = set()
    seen: set[str] = set()
    out: list[str] = []
    collisions = gold_via_collision = unmapped = 0
    for r in results:
        doc = str(r.get("document_id") or "")
        if not doc or doc in seen:
            continue
        seen.add(doc)
        srcs = index.sources.get(doc)
        if not srcs:
            unmapped += 1
            out.append(f"doc:{doc}")
            continue
        pick = next((s for s in srcs if s in gold_set and s not in credited), srcs[0])
        credited.add(pick)
        if len(srcs) > 1 and len(out) < 5:
            collisions += 1
            if pick in gold_set and pick != srcs[0]:
                gold_via_collision += 1
        out.append(pick)
    return Ranking(out, collisions, gold_via_collision, unmapped)
