#!/usr/bin/env bash
# Fetch the public-domain Project Gutenberg seed corpus for `strata eval` / `strata bench`.
# Writes eval/corpus/pg<ID>-<slug>.txt with the Gutenberg licence header/footer stripped
# (keeps only the text between the "*** START OF" and "*** END OF" marker lines).
# Idempotent: existing non-empty files are kept. The corpus is committed (spec §9);
# this script is how it was produced and how to restore it.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=eval/corpus
mkdir -p "$OUT"

# id|slug|substring that must appear in the first 80 lines (guards against a wrong ebook id)
BOOKS=(
  "1342|pride-and-prejudice|Pride and Prejudice"
  "1400|great-expectations|Great Expectations"
  "98|a-tale-of-two-cities|A Tale of Two Cities"
  "2701|moby-dick|Moby"
  "120|treasure-island|Treasure Island"
  "84|frankenstein|Frankenstein"
  "345|dracula|Dracula"
  "1661|the-adventures-of-sherlock-holmes|Sherlock Holmes"
  "1228|on-the-origin-of-species|Origin of Species"
  "944|the-voyage-of-the-beagle|Voyage of the Beagle"
  "5001|relativity|Relativity"
  "14474|chemical-history-of-a-candle|Chemical History of a Candle"
  "1497|the-republic|The Republic"
  "5827|the-problems-of-philosophy|Problems of Philosophy"
  "3300|the-wealth-of-nations|Wealth of Nations"
  "1232|the-prince|The Prince"
)

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

failed=0
for entry in "${BOOKS[@]}"; do
  IFS='|' read -r id slug title <<<"$entry"
  dest="$OUT/pg${id}-${slug}.txt"
  if [[ -s "$dest" ]]; then
    echo "have   $dest"
    continue
  fi
  raw="$tmpdir/pg${id}.txt"
  UA="strata-rag-eval/0.1 (seed corpus fetch)"
  if ! curl -fsSL --retry 3 -A "$UA" -o "$raw" "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"; then
    # Some ebooks (e.g. 5001) have no plain-text edition: convert the HTML edition to text.
    html="$tmpdir/pg${id}.html"
    if ! curl -fsSL --retry 3 -A "$UA" -o "$html" "https://www.gutenberg.org/cache/epub/${id}/pg${id}-images.html"; then
      echo "FAILED download ebook ${id}" >&2; failed=1; continue
    fi
    python3 -I - "$html" >"$raw" <<'PY'
import html.parser, sys
class P(html.parser.HTMLParser):
    BLOCK = {"p","div","br","h1","h2","h3","h4","h5","h6","li","tr","blockquote","pre"}
    def __init__(self):
        super().__init__(convert_charrefs=True); self.out=[]; self.skip=0
    def handle_starttag(self, t, a):
        if t in ("script","style","head"): self.skip += 1
        if t in self.BLOCK: self.out.append("\n\n")
    def handle_endtag(self, t):
        if t in ("script","style","head"): self.skip -= 1
        if t in self.BLOCK: self.out.append("\n\n")
    def handle_data(self, d):
        if not self.skip: self.out.append(" ".join(d.split()) + " ")
p = P(); p.feed(open(sys.argv[1], encoding="utf-8").read())
import re
text = re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(p.out)); text = re.sub(r"\n{3,}", "\n\n", text)
print(text)
PY
  fi
  if ! head -n 80 "$raw" | tr -d '\r' | grep -qiF "$title"; then
    echo "FAILED ebook ${id}: first 80 lines do not mention '${title}' (wrong id?)" >&2
    failed=1; continue
  fi
  tr -d '\r' <"$raw" | awk '
    /^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG/ { on = 0 }
    on { print }
    /^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG/ { on = 1 }
  ' >"$dest.part"
  words=$(wc -w <"$dest.part" | tr -d ' ')
  if [[ "$words" -lt 5000 ]]; then
    echo "FAILED ebook ${id}: only ${words} words between START/END markers" >&2
    rm -f "$dest.part"; failed=1; continue
  fi
  mv "$dest.part" "$dest"
  echo "fetched $dest (${words} words)"
  sleep 2   # be polite to gutenberg.org
done

total=$(cat "$OUT"/*.txt 2>/dev/null | wc -w | tr -d ' ')
echo "corpus: $(ls "$OUT"/*.txt 2>/dev/null | wc -l | tr -d ' ') files, ${total} words"
exit "$failed"
