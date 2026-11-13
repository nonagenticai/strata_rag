# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""End-to-end latency of a LIVE strata-rag API (client-side p50/p95 + the server's timings_ms).

Creates a clearly named synthetic test tree in the caller's tenant (root router
`scale-harness-<run>` with one leaf per book), ingests a small real-text corpus (Gutenberg books
from eval/corpus, split into <= --docs documents; ingest calls the tenant's real embed provider),
runs the questions from eval/questions.jsonl scoped to that root, then deletes the tree
(DELETE /api/stores/{root}?recursive=true) even when a step fails.

  STRATA_API_TOKEN=... uv run scripts/scale_harness/api_bench.py \\
      --base-url http://127.0.0.1:8092 --tenant <uuid> --queries 100

Auth: `Authorization: Bearer $STRATA_API_TOKEN` plus `X-Tenant-Id` (a `*:` token needs it, a
tenant-bound token must match it). A 401/403 (or a tenant without registered providers) is a
result: it is printed and the run stops. The tenant must have embed (+ rerank) providers
registered (PUT /api/tenant/providers); production has no deployment default.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ApiError(Exception):
    def __init__(self, status: int, body: str, what: str):
        super().__init__(f"{what}: HTTP {status}: {body[:300]}")
        self.status = status


def call(a, method: str, path: str, body=None, what: str = ""):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(a.base_url.rstrip("/") + path, data=data, method=method)
    req.add_header("X-Tenant-Id", a.tenant)
    if a.token:
        req.add_header("Authorization", f"Bearer {a.token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            raw = r.read()
            ms = (time.perf_counter() - t0) * 1000
            return (json.loads(raw) if raw else None), ms
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode(errors="replace"), what or f"{method} {path}") from None


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else float("nan")


def corpus(n_docs: int, chars: int) -> list[tuple[str, str, str]]:
    """(book, title, text) documents: each book cut at paragraph boundaries into ~`chars` pieces,
    books interleaved until `n_docs`."""
    books = sorted((ROOT / "eval/corpus").glob("pg*.txt"))
    pieces: dict[str, list[str]] = {}
    for b in books:
        paras, cur, out = b.read_text(errors="replace").split("\n\n"), "", []
        for p in paras:
            cur += p + "\n\n"
            if len(cur) >= chars:
                out.append(cur)
                cur = ""
        pieces[b.name] = out
    docs, i = [], 0
    while len(docs) < n_docs and any(i < len(v) for v in pieces.values()):
        for name, v in pieces.items():
            if i < len(v) and len(docs) < n_docs:
                docs.append((name, f"{name} part {i}", v[i]))
        i += 1
    return docs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--tenant", required=True, help="the synthetic test tenant's uuid")
    p.add_argument("--token", default=os.environ.get("STRATA_API_TOKEN"))
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--doc-chars", type=int, default=6000)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "api.json"))
    a = p.parse_args()

    try:
        _, ms = call(a, "GET", "/health", what="health (auth-exempt)")
        print(f"/health {ms:.0f} ms")
    except (ApiError, urllib.error.URLError) as e:
        sys.exit(f"endpoint unreachable: {e}")

    run = uuid.uuid4().hex[:8]
    root = None
    result: dict = {"base_url": a.base_url, "tenant": a.tenant, "run": run}
    try:
        try:
            r, _ = call(a, "POST", "/api/stores", {"kind": "router", "name": f"scale-harness-{run}",
                                                   "description": "scale harness synthetic test tree; safe to delete"}, "create root")
        except ApiError as e:
            if e.status in (401, 403):
                result["auth"] = str(e)
                print(f"AUTH RESULT: {e}")
                return
            raise
        root = r["id"]
        docs = corpus(a.docs, a.doc_chars)
        leaves = {}
        for book in sorted({d[0] for d in docs}):
            r, _ = call(a, "POST", "/api/stores", {"kind": "leaf", "parent_id": root, "name": book,
                                                   "description": f"Gutenberg text {book} (scale harness)"}, "create leaf")
            leaves[book] = r["id"]
        ing = []
        t0 = time.time()
        for book, title, text in docs:
            _, ms = call(a, "POST", f"/api/stores/{leaves[book]}/documents",
                         {"text": text, "source": f"scale-harness/{book}", "title": title}, "ingest")
            ing.append(ms)
        result["ingest"] = {"docs": len(docs), "seconds": time.time() - t0, "p50_ms": pct(ing, 50), "p95_ms": pct(ing, 95)}
        print(f"ingested {len(docs)} docs in {result['ingest']['seconds']:.0f}s (p50 {pct(ing, 50):.0f} ms)")

        qs = [json.loads(x)["query"] for x in (ROOT / "eval/questions.jsonl").read_text().splitlines() if x.strip()]
        samples = []
        for i in range(a.queries):
            r, ms = call(a, "POST", "/api/query", {"query": qs[i % len(qs)], "top_k": a.top_k, "root_id": root}, "query")
            samples.append({"client_ms": ms, **{k: r["timings_ms"][k] for k in ("embed", "route", "search", "rerank", "total")}})
        result["query"] = {k: {"p50": pct([s[k] for s in samples], 50), "p95": pct([s[k] for s in samples], 95),
                               "mean": statistics.mean(s[k] for s in samples)} for k in samples[0]}
        result["samples"] = samples
        print("| stage | p50 ms | p95 ms |\n|---|---|---|")
        for k, v in result["query"].items():
            print(f"| {k} | {v['p50']:.0f} | {v['p95']:.0f} |")
    except ApiError as e:
        result["error"] = str(e)
        print(f"STOPPED: {e}")
    finally:
        if root:
            try:
                call(a, "DELETE", f"/api/stores/{root}?recursive=true", what="cleanup")
                result["cleanup"] = "deleted"
            except ApiError as e:
                result["cleanup"] = f"FAILED, delete {root} by hand: {e}"
            print(f"cleanup: {result['cleanup']}")
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
