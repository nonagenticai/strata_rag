"""Build ``data/n50.jsonl``: the 50 HotpotQA rows KAG's promoted baseline was scored on.

The question set is defined by KAG's per-row evidence file (``data/kag_n50_20260902.jsonl``,
copied from kag ``eval/baselines_evidence/n50-20260902-RUN35268554903.scores.jsonl``):
its 50 distinct ``example_id`` values. KAG's ``load_hotpotqa(limit=50)`` took them, in order, from
HF ``hotpotqa/hotpot_qa`` config ``distractor`` split ``validation``.

The split is STREAMED (``streaming=True``) and iteration stops as soon as every wanted id is seen,
so the full HotpotQA dump is never downloaded. The output holds the raw rows (id, question, answer,
type, level, context, supporting_facts) in dataset order.

Run (one-off; ``datasets`` is only in the ``fetch`` extra)::

    uv run --extra fetch python fetch_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KAG_ROWS = DATA / "kag_n50_20260902.jsonl"
OUT = DATA / "n50.jsonl"
KEEP = ("id", "question", "answer", "type", "level", "context", "supporting_facts")
# KAG took the first 50 usable rows; a generous cap stops a wrong id set from streaming forever.
MAX_SCAN = 2000


def kag_example_ids(path: Path = KAG_ROWS) -> set[str]:
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["example_id"] for line in fh if line.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build data/n50.jsonl from HF hotpotqa (streamed)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    from datasets import load_dataset  # pyright: ignore[reportMissingImports]

    wanted = kag_example_ids()
    if len(wanted) != 50:
        print(f"expected 50 KAG example ids, found {len(wanted)}", file=sys.stderr)
        return 2

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for raw in ds:
        scanned += 1
        row = dict(raw)
        if row["id"] in wanted:
            rows.append({k: row[k] for k in KEEP})
        if len(rows) == len(wanted) or scanned >= MAX_SCAN:
            break

    got = {r["id"] for r in rows}
    if got != wanted:
        print(
            f"id set mismatch after scanning {scanned} rows: missing={sorted(wanted - got)}",
            file=sys.stderr,
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (scanned {scanned} rows of the split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
