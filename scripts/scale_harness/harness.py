# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.3", "numpy>=2.0", "pgvector>=0.4"]
# ///
"""Scale harness for strata's leaf-filtered dense search (SQL level, synthetic vectors).

Question: with ONE HNSW index over every tenant's chunks and the leaf filter applied to the rows
the index yields, how much recall does a search over a small leaf set lose, and what fixes it
(hnsw.max_scan_tuples, an exact scan below a chunk-count threshold, or per-owner-class partial
HNSW indexes)?

  uv run scripts/scale_harness/harness.py load    --chunks 1000000 --dim 256 --tenants 300
  uv run scripts/scale_harness/harness.py bench   --phase v1 --queries 100
  uv run scripts/scale_harness/harness.py bench   --phase v1mem      # V1 with larger work_mem
  uv run scripts/scale_harness/harness.py partial            # drop the global HNSW, build partials
  uv run scripts/scale_harness/harness.py bench   --phase v2
  uv run scripts/scale_harness/harness.py report
  uv run scripts/scale_harness/harness.py drop

The schema is the repo's migrations/*.sql, applied in order. Deliberate deviations (harness DB
only): `chunks.embedding` is altered to `vector(--dim)` before the load (stores' 1024-d profile
vectors are left as-is); `chunks.owner_class` ('platform'|'tenant') is added for the partial-index
variants; the bulk load runs with `session_replication_role = replica` (no per-row triggers / FK
checks; the generator writes consistent rows). The global HNSW index is built with the exact
statement from migration 0009.

Variants:
  V1  the shipped design: one global HNSW, leaf filter on the rows it yields (DENSE_SQL as-is),
      hnsw.max_scan_tuples default (20000) / 100000 / 1000000; plus "exact" (same SQL, index scans
      off -> bitmap scan of chunks_store_idx, or a seq scan), which is also the ground truth.
  V2  partial HNSW WHERE owner_class='platform' for platform leaf sets; exact scan through the
      store_id btree for tenant leaf sets; mixed = both, merged by distance.
  V3  V2 plus a partial HNSW WHERE owner_class='tenant' for tenant leaf sets.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# SCALE_OUT: a separate results dir per run (e.g. out/dim1024), so one run never overwrites another's.
OUT = Path(os.environ.get("SCALE_OUT", HERE / "out"))
DEFAULT_ADMIN = "postgres://postgres:test@127.0.0.1:5491/postgres"
DB_NAME = "strata_scale"

# Copied verbatim from src/db/search.rs (origin/main 618ddd9). The search runs in its own
# transaction after `set_hnsw_scan`: iterative_scan = relaxed_order, ef_search = 100 (default).
DENSE_SQL = "WITH r AS MATERIALIZED (SELECT id, embedding <=> $1 AS dist FROM chunks WHERE store_id = ANY(ARRAY(SELECT s.id FROM stores s WHERE s.id = ANY($2) AND s.tenant = $4)) ORDER BY embedding <=> $1 LIMIT $3) SELECT id, 1 - dist AS sim FROM r ORDER BY dist, id"
# Mixed tenant + PLATFORM leaves: cross-tenant shared stores are not built yet, so strata has no
# SQL for this. Same statement with the tenant predicate widened to a list (the minimal change a
# shared-store design would make); the chunk scan and its plan are unchanged.
DENSE_SQL_MULTI = DENSE_SQL.replace("s.tenant = $4", "s.tenant = ANY($4)")
# V2/V3: the same statement restricted to one owner class, so the partial index matches.
DENSE_SQL_PLATFORM = DENSE_SQL.replace("FROM chunks WHERE ", "FROM chunks WHERE owner_class = 'platform' AND ")
DENSE_SQL_TENANT = DENSE_SQL.replace("FROM chunks WHERE ", "FROM chunks WHERE owner_class = 'tenant' AND ")
# V1 "forced HNSW": what the shipped statement does whenever the planner picks the HNSW index
# (it decides per statement from estimates; it never sees the leaf set's real size because the
# filter is an ARRAY(subquery)). The coalesce() hides store_id from chunks_store_idx so the
# planner cannot take the btree; with seq scans off the HNSW index is the only ordered path.
SET_HNSW_SQL = "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), set_config('hnsw.ef_search', %s, true)"
CANDIDATES = 50  # src/search.rs CANDIDATES: the LIMIT every dense search runs with
EF_SEARCH = 100  # src/db/search.rs DEFAULT_EF_SEARCH
MST = {"mst_default": None, "mst_100k": 100_000, "mst_1m": 1_000_000}
# Phase v1mem: the iterative scan also stops when its memory reaches work_mem ×
# hnsw.scan_mem_multiplier (4MB × 1 by default), which at 1M rows binds long before
# max_scan_tuples does. Same statement, max_scan_tuples 1M, larger work_mem; capped by
# statement_timeout (a timeout counts as 0 rows at the timeout's latency).
# 16MB = the production primary's work_mem.
MEM = {"wm16MB": "16MB", "wm64MB": "64MB", "wm256MB": "256MB"}
MEM_TIMEOUT_MS = 5000


def db_url(admin: str) -> str:
    return admin.rsplit("/", 1)[0] + "/" + DB_NAME


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def db_size(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]


def chunk_vectors(rng: np.random.Generator, center: np.ndarray, n: int, sigma: float) -> np.ndarray:
    g = rng.standard_normal((n, center.shape[0])).astype(np.float32)
    return unit(center.astype(np.float32) + sigma * g / np.sqrt(center.shape[0])).astype(np.float32)


# --------------------------------------------------------------------------------------- corpus


def build_corpus(a: argparse.Namespace, rng: np.random.Generator) -> tuple[list, list[dict], np.ndarray]:
    """PLATFORM tree (base → industries → departments → jobs) holding most of the volume, plus N
    tenants each with an org tree (org leaf, departments, jobs) whose leaves mirror popular
    platform jobs. Every leaf's cluster center derives from its parent's, so siblings are close and
    a tenant's knowledge sits right next to the platform's (and other tenants') knowledge about the
    same job: the hard case for a post-filtered ANN scan. A few explicitly sized "big" tenants
    give the exact-vs-HNSW crossover for a single tenant's large leaf set."""
    d = a.dim
    rnd = lambda: unit(rng.standard_normal(d))  # noqa: E731
    child = lambda c, s: unit(c + s * rnd())  # noqa: E731

    routers: list[tuple] = []  # (id, tenant, parent, name)
    leaves: list[dict] = []
    centers: list[np.ndarray] = []

    def leaf(tenant, parent, name, role, center, **kw) -> dict:
        l = {"id": str(uuid.uuid4()), "tenant": tenant, "parent": parent, "name": name, "role": role,
             "i": len(leaves), "chunks": 0} | kw
        leaves.append(l)
        centers.append(center)
        return l

    plat = str(uuid.uuid4())
    root = str(uuid.uuid4())
    routers.append((root, plat, None, "platform"))
    base = leaf(plat, root, "base", "base", rnd())
    jobs, depts, inds = [], [], []
    for i in range(a.industries):
        ir = str(uuid.uuid4())
        routers.append((ir, plat, root, f"industry-{i}"))
        ic = child(centers[base["i"]], 1.5)
        inds.append(leaf(plat, ir, "general", "industry", ic))
        for j in range(a.departments):
            dr = str(uuid.uuid4())
            routers.append((dr, plat, ir, f"dept-{j}"))
            dc = child(ic, 1.0)
            depts.append(leaf(plat, dr, "general", "dept", dc))
            for k in range(a.jobs):
                jobs.append(leaf(plat, dr, f"job-{k}", "job", child(dc, 0.7),
                                 pj=len(jobs), dept=depts[-1]["i"], ind=inds[-1]["i"]))

    popular = rng.choice(len(jobs), size=min(a.popular_jobs, len(jobs)), replace=False)
    big = [int(x) for x in a.big_tenants.split(",") if x]
    tenants: list[list[dict]] = []
    for t in range(a.tenants + len(big)):
        tid = str(uuid.uuid4())
        tr = str(uuid.uuid4())
        routers.append((tr, tid, None, "org"))
        pjs = [int(x) for x in rng.choice(popular, size=a.tenant_departments * a.tenant_jobs)]
        mine = [leaf(tid, tr, "org", "t_org", child(centers[jobs[pjs[0]]["ind"]], 0.7))]
        for j in range(a.tenant_departments):
            dr = str(uuid.uuid4())
            routers.append((dr, tid, tr, f"dept-{j}"))
            first = jobs[pjs[j * a.tenant_jobs]]
            dl = leaf(tid, dr, "general", "t_dept", child(centers[first["dept"]], 0.5), pj=first["pj"])
            mine.append(dl)
            for k in range(a.tenant_jobs):
                p = jobs[pjs[j * a.tenant_jobs + k]]
                mine.append(leaf(tid, dr, f"job-{k}", "t_job", child(centers[p["i"]], 0.5),
                                 pj=p["pj"], dept_leaf=dl["i"], org=mine[0]["i"]))
        tenants.append(mine)

    # Volume: --platform-share of --chunks in the PLATFORM tree (base 5% / industry 20% / dept 35%
    # / job 40%); big tenants get their stated sizes; the rest across the regular tenants
    # (lognormal tenant sizes), each org 30% / dept 30% / job 40%. Leaf sizes jittered lognormally.
    plat_total = int(a.chunks * a.platform_share)
    reg_total = a.chunks - plat_total - sum(big)
    if reg_total < a.tenants:
        sys.exit("--big-tenants leave no volume for the regular tenants")

    def spread(ls: list[dict], budget: int) -> None:
        w = rng.lognormal(0.0, 0.5, len(ls))
        for l, c in zip(ls, np.maximum(1, np.floor(budget * w / w.sum())).astype(int)):
            l["chunks"] = int(c)

    spread([base], int(plat_total * 0.05))
    spread(inds, int(plat_total * 0.20))
    spread(depts, int(plat_total * 0.35))
    spread(jobs, int(plat_total * 0.40))
    tw = rng.lognormal(0.0, 0.8, a.tenants)
    budgets = [int(reg_total * w / tw.sum()) for w in tw] + big
    for ti, (mine, tb) in enumerate(zip(tenants, budgets)):
        for role, share in (("t_org", 0.3), ("t_dept", 0.3), ("t_job", 0.4)):
            spread([l for l in mine if l["role"] == role], max(1, int(tb * share)))
        for l in mine:
            l["big"] = tb if ti >= a.tenants else None
    return routers, leaves, np.stack(centers).astype(np.float32)


def cmd_load(a: argparse.Namespace) -> None:
    # heap row (vector + uuids + short text/tsv + header) + global HNSW tuple + btree/gin share.
    # The partial-index phase drops the global index before building partials of the same total size.
    est = a.chunks * ((a.dim * 4 + 200) + (a.dim * 4 + 130) + 90)
    print(f"estimated DB size with one HNSW: {gb(est)} (budget {a.max_db_gb} GB)")
    if est > a.max_db_gb * 1e9:
        sys.exit("refusing: estimate exceeds --max-db-gb; lower --chunks or --dim")
    with psycopg.connect(a.admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {DB_NAME}")
    with psycopg.connect(db_url(a.admin), autocommit=True) as conn:
        migs = sorted((ROOT / "migrations").glob("[0-9]*_*.sql"))
        for f in migs:
            with conn.transaction():
                conn.execute(f.read_text())
        print(f"applied {len(migs)} migrations ({migs[0].name} .. {migs[-1].name})")
        conn.execute("DROP INDEX chunks_embedding_hnsw")
        conn.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({a.dim})")
        conn.execute("ALTER TABLE chunks ADD COLUMN owner_class text CHECK (owner_class IN ('platform','tenant'))")
        register_vector(conn)

        rng = np.random.default_rng(a.seed)
        routers, leaves, centers = build_corpus(a, rng)
        plat = leaves[0]["tenant"]
        zero = "[" + ",".join(["0"] * 1024) + "]"
        conn.execute("SET session_replication_role = replica")
        with conn.cursor() as cur, cur.copy(
            "COPY stores (id, tenant, parent_id, kind, name, description, desc_vec, profile_vec, embed_model, chunk_count) FROM STDIN"
        ) as cp:
            for rid, ten, par, name in routers:
                cp.write_row((rid, ten, par, "router", name, name, zero, zero, "synthetic", 0))
            for l in leaves:
                cp.write_row((l["id"], l["tenant"], l["parent"], "leaf", l["name"], l["role"], zero, zero, "synthetic", l["chunks"]))
        total = sum(l["chunks"] for l in leaves)
        n_ten = len({l["tenant"] for l in leaves}) - 1
        print(f"stores: {len(routers)} routers, {len(leaves)} leaves; {n_ten} tenants + PLATFORM; {total} chunks planned")

        t0 = time.time()
        done = 0
        docs: dict[int, list[uuid.UUID]] = {}
        with conn.cursor() as cur, cur.copy("COPY documents (id, store_id, source, content_hash, token_count) FROM STDIN (FORMAT BINARY)") as dcp:
            dcp.set_types(["uuid", "uuid", "text", "bytea", "int4"])
            for l in leaves:
                docs[l["i"]] = [uuid.uuid4() for _ in range(0, l["chunks"], a.chunks_per_doc)]
                for k, doc in enumerate(docs[l["i"]]):
                    dcp.write_row((doc, uuid.UUID(l["id"]), f"synthetic/{l['i']}/{k}", os.urandom(32), 400))
        with conn.cursor() as cur, cur.copy("COPY chunks (id, document_id, store_id, ord, text, token_count, embedding, owner_class) FROM STDIN (FORMAT BINARY)") as ccp:
            ccp.set_types(["uuid", "uuid", "uuid", "int4", "text", "int4", "vector", "text"])
            for l in leaves:
                vecs = chunk_vectors(rng, centers[l["i"]], l["chunks"], a.sigma)
                sid = uuid.UUID(l["id"])
                owner = "platform" if l["tenant"] == plat else "tenant"
                for i in range(l["chunks"]):
                    ccp.write_row((uuid.uuid4(), docs[l["i"]][i // a.chunks_per_doc], sid, i % a.chunks_per_doc,
                                   f"{l['role']} {l['i']} chunk {i}", 400, vecs[i], owner))
                done += l["chunks"]
                if l["i"] % 500 == 0:
                    print(f"  {done}/{total} chunks, {time.time() - t0:.0f}s", flush=True)
        conn.execute("RESET session_replication_role")
        load_s = time.time() - t0
        print(f"loaded {done} chunks in {load_s:.0f}s; DB {gb(db_size(conn))}")
        conn.execute("VACUUM ANALYZE")
        heap = db_size(conn)
        print(f"after VACUUM ANALYZE: {gb(heap)}")

        hnsw = next(x for x in (ROOT / "migrations/0009_chunks_hnsw.sql").read_text().splitlines() if x.startswith("CREATE INDEX"))
        conn.execute(f"SET maintenance_work_mem = '{a.build_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers = {a.build_workers}")
        t0 = time.time()
        conn.execute(hnsw)
        build_s = time.time() - t0
        conn.execute("ANALYZE chunks")
        size = db_size(conn)
        idx = conn.execute("SELECT pg_relation_size('chunks_embedding_hnsw')").fetchone()[0]
        print(f"global HNSW built in {build_s:.0f}s ({gb(idx)}); DB {gb(size)}")

    state = {"args": {k: v for k, v in vars(a).items() if k != "admin"}, "chunks": done, "load_s": load_s,
             "build_s": build_s, "db_bytes_no_index": heap, "db_bytes": size, "hnsw_bytes": idx,
             "platform": plat, "leaves": [{k: v for k, v in l.items() if k != "parent"} for l in leaves]}
    np.save(OUT / "centers.npy", centers)
    (OUT / "state.json").write_text(json.dumps(state))


def cmd_partial(a: argparse.Namespace) -> None:
    """V2/V3 indexes: drop the global HNSW (disk budget), build one partial HNSW per owner class
    with the migration's opclass and default build parameters."""
    state = json.loads((OUT / "state.json").read_text())
    with psycopg.connect(db_url(a.admin), autocommit=True) as conn:
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        conn.execute(f"SET maintenance_work_mem = '{a.build_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers = {a.build_workers}")
        for cls in ("platform", "tenant"):
            t0 = time.time()
            conn.execute(f"CREATE INDEX chunks_embedding_hnsw_{cls} ON chunks USING hnsw (embedding vector_cosine_ops) WHERE owner_class = '{cls}'")
            s = time.time() - t0
            sz = conn.execute(f"SELECT pg_relation_size('chunks_embedding_hnsw_{cls}')").fetchone()[0]
            state[f"partial_{cls}_build_s"], state[f"partial_{cls}_bytes"] = s, sz
            print(f"partial HNSW {cls}: {s:.0f}s, {gb(sz)}", flush=True)
        conn.execute("ANALYZE chunks")
        state["db_bytes_partials"] = db_size(conn)
        print(f"DB {gb(state['db_bytes_partials'])}")
    (OUT / "state.json").write_text(json.dumps(state))


# ---------------------------------------------------------------------------------------- bench


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def psy(sql: str, params: tuple) -> tuple[str, dict]:
    """strata's $n statements -> psycopg named placeholders (same server-side parameters)."""
    for n in range(len(params), 0, -1):
        sql = sql.replace(f"${n}", f"%(p{n})s")
    return sql, {f"p{i + 1}": v for i, v in enumerate(params)}


def plan_kind(conn: psycopg.Connection, sql: str, params: tuple) -> str:
    sql, params = psy(sql, params)
    plan = json.dumps(conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()[0])
    for idx, k in (("chunks_embedding_hnsw_platform", "hnsw_platform"), ("chunks_embedding_hnsw_tenant", "hnsw_tenant"),
                   ("chunks_embedding_hnsw", "hnsw"), ("chunks_store_idx", "store_idx")):
        if idx in plan:
            return k
    return "seqscan" if "Seq Scan" in plan else "other"


def make_cases(a: argparse.Namespace, state: dict, centers: np.ndarray) -> list[dict]:
    """Leaf sets an agent would search, and one query vector per case drawn from the target leaf's
    cluster. tenant_scope = a tenant job agent's own leaves (org + dept + job); platform_job = one
    platform job leaf; platform_scope = base + industry + dept + job of the platform; mixed =
    tenant_scope + the platform_scope of the job it mirrors; tenant_big = every leaf of a big
    tenant (the large single-tenant case)."""
    L = state["leaves"]
    plat = state["platform"]
    rng = np.random.default_rng(a.seed + 1)
    jobs = [l for l in L if l["role"] == "job"]
    t_jobs = [l for l in L if l["role"] == "t_job" and not l.get("big")]
    base = next(l for l in L if l["role"] == "base")
    pscope = lambda j: [base, L[j["ind"]], L[j["dept"]], j]  # noqa: E731
    tscope = lambda tj: [L[tj["org"]], L[tj["dept_leaf"]], tj]  # noqa: E731
    cases = []

    def add(kind, ls, tenants, target):
        q = chunk_vectors(rng, centers[target["i"]], 1, a.sigma)[0]
        cases.append({"set": kind, "leaves": [l["id"] for l in ls], "tenants": tenants,
                      "classes": sorted({"platform" if l["tenant"] == plat else "tenant" for l in ls}),
                      "n_chunks": sum(l["chunks"] for l in ls), "q": q.tolist()})

    for _ in range(a.queries):
        tj = t_jobs[rng.integers(len(t_jobs))]
        add("tenant_scope", tscope(tj), [tj["tenant"]], tj)
    for _ in range(a.queries):
        j = jobs[rng.integers(len(jobs))]
        add("platform_job", [j], [plat], j)
    for _ in range(a.queries):
        j = jobs[rng.integers(len(jobs))]
        add("platform_scope", pscope(j), [plat], j)
    for _ in range(a.queries):
        tj = t_jobs[rng.integers(len(t_jobs))]
        add("mixed", tscope(tj) + pscope(jobs[tj["pj"]]), [tj["tenant"], plat], tj)
    for size in sorted({l["big"] for l in L if l.get("big")}):
        mine = [l for l in L if l.get("big") == size]
        tjs = [l for l in mine if l["role"] == "t_job"]
        for _ in range(max(10, a.queries // 2)):
            add(f"tenant_big_{size}", mine, [mine[0]["tenant"]], tjs[rng.integers(len(tjs))])
    return cases


def run(conn, sql, params, *, mst=None, hnsw=True, force=False, work_mem=None, explain_to: Path | None = None, label=""):
    """One search in its own transaction, as strata runs it. Returns (ids, ms, plan)."""
    if work_mem is not None:
        try:
            with conn.transaction():
                conn.execute("SELECT set_config('work_mem', %s, true), set_config('statement_timeout', %s, true)", (work_mem, str(MEM_TIMEOUT_MS)))
                return run(conn, sql, params, mst=mst, explain_to=explain_to, label=label)
        except psycopg.errors.QueryCanceled:
            return [], float(MEM_TIMEOUT_MS), "timeout"
    with conn.transaction():
        if hnsw:
            conn.execute(SET_HNSW_SQL, (str(EF_SEARCH),))
            if mst is not None:
                conn.execute("SELECT set_config('hnsw.max_scan_tuples', %s, true)", (str(mst),))
            if force:
                conn.execute("SET LOCAL enable_seqscan = off")
                sql = sql.replace("store_id = ANY(", "coalesce(store_id, store_id) = ANY(")
        else:
            conn.execute("SET LOCAL enable_indexscan = off")  # exact: bitmap on chunks_store_idx / seq scan
        plan = plan_kind(conn, sql, params)
        sql, params = psy(sql, params)
        t0 = time.perf_counter()
        got = conn.execute(sql, params, prepare=True).fetchall()
        ms = (time.perf_counter() - t0) * 1000
        if explain_to is not None and not explain_to.exists():
            txt = "\n".join(r[0] for r in conn.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, params).fetchall())
            explain_to.write_text(f"-- {label}\n{txt}\n")
    return [(r[0], 1 - r[1]) for r in got], ms, plan


def params_for(case: dict, tenants: list[str] | None = None) -> tuple[str, tuple]:
    q = np.array(case["q"], dtype=np.float32)
    ids = [uuid.UUID(x) for x in case["leaves"]]
    ts = tenants if tenants is not None else case["tenants"]
    if len(ts) == 1:
        return DENSE_SQL, (q, ids, CANDIDATES, uuid.UUID(ts[0]))
    return DENSE_SQL_MULTI, (q, ids, CANDIDATES, [uuid.UUID(t) for t in ts])


def class_query(case: dict, cls: str, plat: str) -> tuple[str, tuple]:
    """The case's search restricted to one owner class (matching that class's partial index):
    platform -> the PLATFORM tenant's leaves, tenant -> the calling tenant's leaves."""
    sql, p = params_for(case, [plat] if cls == "platform" else [t for t in case["tenants"] if t != plat])
    return (DENSE_SQL_PLATFORM if cls == "platform" else DENSE_SQL_TENANT), p


def score(got: list, truth: list[str], expect: int) -> dict:
    g = [str(x[0]) for x in got]
    return {"rows": len(g), "short": len(g) < expect,
            "r10": len(set(g[:10]) & set(truth[:10])) / max(1, min(10, len(truth))),
            "r50": len(set(g) & set(truth)) / max(1, len(truth))}


def cmd_bench(a: argparse.Namespace) -> None:
    state = json.loads((OUT / "state.json").read_text())
    centers = np.load(OUT / "centers.npy")
    exp = OUT / "explain"
    exp.mkdir(exist_ok=True)
    plat = state["platform"]
    if a.phase == "v1":
        cases = make_cases(a, state, centers)
    elif a.phase == "v1mem":
        cases = json.loads((OUT / "results_v1.json").read_text())
        with psycopg.connect(db_url(a.admin)) as conn:
            register_vector(conn)
            for qi, c in enumerate(cases):
                sql, p = params_for(c)
                for v, wm in MEM.items():
                    if f"V1_{v}" in c:
                        continue
                    got, ms, plan = run(conn, sql, p, mst=1_000_000, work_mem=wm, explain_to=exp / f"{c['set']}-V1-{v}.txt",
                                        label=f"{c['set']}: {c['n_chunks']} chunks, V1 global HNSW, mst 1M, work_mem {wm}")
                    c[f"V1_{v}"] = {"ms": ms, "plan": plan} | score(got, c["truth"], min(CANDIDATES, c["n_chunks"]))
                if qi % 50 == 0:
                    print(f"  {qi}/{len(cases)} {c['set']} n={c['n_chunks']} " + " ".join(f"{v}:{c[f'V1_{v}']['ms']:.0f}ms/{c[f'V1_{v}']['rows']}r" for v in MEM), flush=True)
        (OUT / "results_v1.json").write_text(json.dumps(cases))
        return cmd_report(a)
    else:
        cases = json.loads((OUT / "results_v1.json").read_text())
    with psycopg.connect(db_url(a.admin)) as conn:
        register_vector(conn)
        if a.phase == "v2":
            # compared against V1_wm16MB: the production primary's work_mem (scan memory budget)
            conn.execute("SET work_mem = '16MB'")
            conn.commit()
        for _ in range(3):  # warm-up: page in the index + heap (the server's shared_buffers is small)
            for c in cases[:: max(1, len(cases) // 40)]:
                sql, p = params_for(c)
                if a.phase == "v1":
                    run(conn, sql, p)
                else:
                    for cls in c["classes"]:
                        run(conn, *class_query(c, cls, plat))
        for qi, c in enumerate(cases):
            expect = min(CANDIDATES, c["n_chunks"])
            sql, p = params_for(c)
            ex = lambda v: exp / f"{c['set']}-{v}.txt"  # noqa: E731
            lbl = lambda v: f"{c['set']}: {len(c['leaves'])} leaves, {c['n_chunks']} chunks, {v}"  # noqa: E731
            if a.phase == "v1":
                truth, ms, plan = run(conn, sql, p, hnsw=False, explain_to=ex("exact"), label=lbl("exact (enable_indexscan=off)"))
                c["truth"] = [str(x[0]) for x in truth]
                c["exact"] = {"ms": ms, "plan": plan} | score(truth, c["truth"], expect)
                for v, mst in MST.items():
                    got, ms, plan = run(conn, sql, p, mst=mst, explain_to=ex(f"V1-{v}"), label=lbl(f"V1 global HNSW, {v}"))
                    c[f"V1_{v}"] = {"ms": ms, "plan": plan} | score(got, c["truth"], expect)
                    got, ms, plan = run(conn, sql, p, mst=mst, force=True, explain_to=ex(f"V1forced-{v}"), label=lbl(f"V1 global HNSW forced, {v}"))
                    c[f"V1forced_{v}"] = {"ms": ms, "plan": plan} | score(got, c["truth"], expect)
            else:
                # V2: platform part -> partial platform HNSW; tenant part -> exact through store_id.
                # V3: tenant part -> partial tenant HNSW instead. Mixed = both parts merged by
                # distance (run sequentially here; latency is the sum of the parts).
                parts: dict[str, tuple] = {}
                if "platform" in c["classes"]:
                    parts["p"] = run(conn, *class_query(c, "platform", plat), explain_to=ex("V2-platform"), label=lbl("partial platform HNSW"))
                if "tenant" in c["classes"]:
                    tsql, tp = class_query(c, "tenant", plat)
                    parts["t_exact"] = run(conn, params_for(c, [t for t in c["tenants"] if t != plat])[0], tp, hnsw=False,
                                           explain_to=ex("V2-tenant-exact"), label=lbl("tenant part, exact via chunks_store_idx"))
                    # forced: left to itself the planner takes chunks_store_idx here (= V2)
                    parts["t_hnsw"] = run(conn, tsql, tp, force=True, explain_to=ex("V3-tenant"), label=lbl("partial tenant HNSW (forced)"))
                for v, tkey in (("V2", "t_exact"), ("V3", "t_hnsw")):
                    use = [parts[k] for k in ("p", tkey) if k in parts]
                    merged = sorted((x for u in use for x in u[0]), key=lambda x: (x[1], str(x[0])))[:CANDIDATES]
                    c[v] = {"ms": sum(u[1] for u in use), "plan": "+".join(u[2] for u in use)} | score(merged, c["truth"], expect)
                    c[f"{v}_parts_ms"] = {k: parts[k][1] for k in ("p", tkey) if k in parts}
            if qi % 50 == 0:
                keys = [k for k in c if k.startswith("V") and isinstance(c[k], dict) and "ms" in c[k]]
                print(f"  {qi}/{len(cases)} {c['set']} n={c['n_chunks']} "
                      + " ".join(f"{k}:{c[k]['ms']:.1f}ms/{c[k]['rows']}r/r10={c[k]['r10']:.2f}/{c[k]['plan']}" for k in keys), flush=True)
    (OUT / f"results_{a.phase}.json").write_text(json.dumps(cases))
    cmd_report(a)


# --------------------------------------------------------------------------------------- report


def cmd_report(a: argparse.Namespace) -> None:
    f1, f2 = OUT / "results_v1.json", OUT / "results_v2.json"
    cases = json.loads((f2 if f2.exists() else f1).read_text())
    state = json.loads((OUT / "state.json").read_text())
    sets = list(dict.fromkeys(c["set"] for c in cases))
    paths = ["exact"] + [f"V1_{v}" for v in MST] + [f"V1forced_{v}" for v in MST] + [f"V1_{v}" for v in MEM] + [p for p in ("V2", "V3") if p in cases[0]]
    out = [f"chunks={state['chunks']} dim={state['args']['dim']} tenants={state['args']['tenants']}+big({state['args']['big_tenants']}) "
           f"load={state['load_s']:.0f}s global_hnsw_build={state['build_s']:.0f}s hnsw={gb(state['hnsw_bytes'])} db={gb(state['db_bytes'])}"
           + (f" partial_platform={state['partial_platform_build_s']:.0f}s/{gb(state['partial_platform_bytes'])}"
              f" partial_tenant={state['partial_tenant_build_s']:.0f}s/{gb(state['partial_tenant_bytes'])}" if "partial_platform_bytes" in state else ""),
           "", "| set | n | chunks in set (median) | path | plan | p50 ms | p95 ms | recall@10 mean | recall@10 min | recall@50 mean | short (rows < min(50, set)) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sets:
        rs = [c for c in cases if c["set"] == s]
        med = int(statistics.median(c["n_chunks"] for c in rs))
        for pth in paths:
            if pth not in rs[0]:
                continue
            ms = [c[pth]["ms"] for c in rs]
            plans = ",".join(f"{p}:{sum(c[pth]['plan'] == p for c in rs)}" for p in sorted({c[pth]["plan"] for c in rs}))
            out.append(f"| {s} | {len(rs)} | {med} | {pth} | {plans} | {pct(ms, 50):.1f} | {pct(ms, 95):.1f} | "
                       f"{statistics.mean(c[pth]['r10'] for c in rs):.3f} | {min(c[pth]['r10'] for c in rs):.3f} | "
                       f"{statistics.mean(c[pth]['r50'] for c in rs):.3f} | {sum(c[pth]['short'] for c in rs)} |")
    # Threshold policy on V1: exact when the leaf set's chunk count (sum of stores.chunk_count,
    # known before the query) is below T, else the global HNSW with the given max_scan_tuples.
    out += ["", "| policy (all sets above) | T (chunks) | p50 ms | p95 ms | recall@10 mean | recall@10 min | short |", "|---|---|---|---|---|---|---|"]
    for v in [f"V1_{m}" for m in MST] + [f"V1_{m}" for m in MEM if f"V1_{m}" in cases[0]]:
        for T in [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 10**12]:
            pick = [c["exact"] if c["n_chunks"] < T else c[v] for c in cases]
            ms = [x["ms"] for x in pick]
            tl = "all exact" if T == 10**12 else ("never" if T == 0 else str(T))
            out.append(f"| exact if < T else {v} | {tl} | {pct(ms, 50):.1f} | {pct(ms, 95):.1f} | "
                       f"{statistics.mean(x['r10'] for x in pick):.3f} | {min(x['r10'] for x in pick):.3f} | {sum(x['short'] for x in pick)} |")
    # Exact-vs-HNSW crossover on tenant-only leaf sets, by chunk-count bucket.
    tcases = [c for c in cases if c["classes"] == ["tenant"]]
    if tcases:
        out += ["", "| tenant-only leaf sets, chunks in set | n | exact p50 | exact p95 | V1 forced HNSW p50 (mst default) | V1 forced r10 | V1 forced mst_1m p50 | r10 | V3 tenant HNSW p50 | V3 r10 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        edges = [0, 200, 1_000, 5_000, 20_000, 50_000, 200_000, 10**9]
        for lo, hi in zip(edges, edges[1:]):
            b = [c for c in tcases if lo <= c["n_chunks"] < hi]
            if not b:
                continue
            row = [f"{lo}-{hi if hi < 10**9 else 'inf'}", str(len(b)), f"{pct([c['exact']['ms'] for c in b], 50):.1f}", f"{pct([c['exact']['ms'] for c in b], 95):.1f}",
                   f"{pct([c['V1forced_mst_default']['ms'] for c in b], 50):.1f}", f"{statistics.mean(c['V1forced_mst_default']['r10'] for c in b):.3f}",
                   f"{pct([c['V1forced_mst_1m']['ms'] for c in b], 50):.1f}", f"{statistics.mean(c['V1forced_mst_1m']['r10'] for c in b):.3f}"]
            if "V3" in b[0]:
                row += [f"{pct([c['V3_parts_ms']['t_hnsw'] for c in b], 50):.1f}", f"{statistics.mean(c['V3']['r10'] for c in b):.3f}"]
            else:
                row += ["-", "-"]
            out.append("| " + " | ".join(row) + " |")
    txt = "\n".join(out)
    (OUT / "tables.md").write_text(txt + "\n")
    print(txt)


def cmd_drop(a: argparse.Namespace) -> None:
    with psycopg.connect(a.admin, autocommit=True) as c:
        before = int(c.execute("SELECT sum(pg_database_size(datname)) FROM pg_database").fetchone()[0])
        c.execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")
        after = int(c.execute("SELECT sum(pg_database_size(datname)) FROM pg_database").fetchone()[0])
    print(f"dropped {DB_NAME}: databases on the server {gb(before)} -> {gb(after)} (freed {gb(before - after)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--admin", default=os.environ.get("SCALE_ADMIN_URL", DEFAULT_ADMIN),
                   help="URL of an existing database on the target server; strata_scale is created beside it")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sigma", type=float, default=0.8, help="chunk/query noise around a leaf center")
    p.add_argument("--build-mem", default="2GB", help="maintenance_work_mem for index builds")
    p.add_argument("--build-workers", type=int, default=0, help="parallel HNSW build workers; they share memory via /dev/shm, which is 64MB in a default docker container, so the default is a serial build")
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("load")
    l.add_argument("--chunks", type=int, default=1_000_000)
    l.add_argument("--dim", type=int, default=256)
    l.add_argument("--tenants", type=int, default=300, help="regular (small) tenants")
    l.add_argument("--big-tenants", default="2000,8000,25000,60000", help="chunk counts of extra, explicitly sized tenants")
    l.add_argument("--platform-share", type=float, default=0.85)
    l.add_argument("--industries", type=int, default=20)
    l.add_argument("--departments", type=int, default=8)
    l.add_argument("--jobs", type=int, default=6)
    l.add_argument("--tenant-departments", type=int, default=3)
    l.add_argument("--tenant-jobs", type=int, default=3)
    l.add_argument("--popular-jobs", type=int, default=60, help="platform jobs that tenants' jobs mirror")
    l.add_argument("--chunks-per-doc", type=int, default=20)
    l.add_argument("--max-db-gb", type=float, default=4.5)
    b = sub.add_parser("bench")
    b.add_argument("--phase", choices=["v1", "v1mem", "v2"], required=True, help="v1: global HNSW + exact truth; v2: partial indexes (run `partial` first)")
    b.add_argument("--queries", type=int, default=100, help="per leaf-set kind (v1 only; v2 replays v1's cases)")
    sub.add_parser("partial")
    sub.add_parser("report")
    sub.add_parser("drop")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    {"load": cmd_load, "bench": cmd_bench, "partial": cmd_partial, "report": cmd_report, "drop": cmd_drop}[a.cmd](a)


if __name__ == "__main__":
    main()
