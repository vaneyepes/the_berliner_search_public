#!/usr/bin/env python3
"""
Run semantic search for a list of queries and generate:
- raw_results.tsv  (verbatim CLI output for provenance)
- labels.tsv       (with preview text from metadata.jsonl so you can label)

Usage:
  python eval_scripts/run_eval.py \
    --model "sentence-transformers/multi-qa-mpnet-base-dot-v1" \
    --queries queries.txt \
    --out eval/mpnet -k 10
"""

from __future__ import annotations
import argparse, subprocess, shlex, re
from pathlib import Path
from datetime import datetime
import json

# Matches lines printed by the CLI like:
# "01    score=0.8123    chunk=TB_2024_11#c_0061"
LINE_RE = re.compile(r"^\s*(\d+)\s*score=([0-9.]+)\s*chunk=([^\s]+)")

def run_search(query: str, model: str, k: int) -> str:
    cmd = f'python -m berliner.cli search {shlex.quote(query)} --model {shlex.quote(model)} -k {k}'
    cp = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if cp.returncode != 0:
        return f"# ERROR for query: {query}\n{cp.stderr.strip()}\n"
    return cp.stdout

def build_preview_map(meta_path: Path, max_words: int = 80) -> dict[str, str]:
    """Load data/enriched/metadata.jsonl and build chunk_id -> short snippet."""
    out: dict[str, str] = {}
    if not meta_path.exists():
        return out
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("chunk_id")
        txt = (rec.get("summary_text") or rec.get("chunk_text") or "").strip()
        if cid and txt and cid not in out:
            words = txt.split()
            out[cid] = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id used to build the index.")
    ap.add_argument("--queries", default="queries.txt", help="Path to queries.txt")
    ap.add_argument("--out", required=True, help="Output folder, e.g., eval/mpnet")
    ap.add_argument("-k", type=int, default=10, help="Top-K to retrieve")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [q.strip() for q in Path(args.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = out_dir / "raw_results.tsv"
    labels_path = out_dir / "labels.tsv"

    # 1) Run searches and write raw results (for provenance)
    with raw_path.open("w", encoding="utf-8") as raw:
        raw.write(f"# model\t{args.model}\n# k\t{args.k}\n# queries\t{args.queries}\n# time\t{timestamp}\n")
        for q in queries:
            raw.write(f"\n## QUERY\t{q}\n")
            raw.write(run_search(q, args.model, args.k))

    # 2) Build preview map from enriched metadata
    preview_map = build_preview_map(Path("data/enriched/metadata.jsonl"))

    # 3) Parse raw and write labels.tsv WITH previews
    rows = []
    current_q = None
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## QUERY"):
            current_q = line.split("\t", 1)[-1].strip()
            continue
        m = LINE_RE.match(line)
        if not m or not current_q:
            continue
        rank = m.group(1)
        score = m.group(2)
        chunk_id = m.group(3)
        preview = preview_map.get(chunk_id, "")
        rows.append((current_q, rank, score, chunk_id, preview))

    with labels_path.open("w", encoding="utf-8") as lab:
        lab.write("query\trank\tscore\tchunk_id\tpreview\tis_relevant\tis_ad\tnotes\n")
        for (q, r, s, cid, prev) in rows:
            # tabs in preview would break TSV; replace with spaces
            safe_prev = prev.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            lab.write(f"{q}\t{r}\t{s}\t{cid}\t{safe_prev}\t\t\t\n")

    print(f"[eval] wrote {raw_path}")
    print(f"[eval] wrote {labels_path}")
    print("[eval] Open labels.tsv and fill 'is_relevant' with Y/N (blank = not relevant). 'is_ad' optional Y/N.")

if __name__ == "__main__":
    main()
