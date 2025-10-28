#!/usr/bin/env python3
"""
Score labels.tsv and print per-query and macro P@10 + ad counts.

Usage:
  python eval_scripts/score_eval.py eval/mpnet/labels.tsv
"""
from __future__ import annotations
import sys, csv, json
from collections import defaultdict
from pathlib import Path

def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        need = {"query", "rank", "is_relevant"}  # preview is optional
        miss = need - set(rdr.fieldnames or [])
        if miss:
            raise SystemExit(f"[score] labels.tsv header missing columns: {', '.join(sorted(miss))}")
        for row in rdr:
            yield row

def as_bool(val: str) -> bool:
    v = (val or "").strip().upper()
    return v.startswith("Y")  # Y / YES / y

def p_at_10(rows):
    rs = []
    for r in rows:
        try:
            rank_int = int(r["rank"])
        except Exception:
            continue
        rs.append((rank_int, as_bool(r.get("is_relevant"))))
    rs.sort(key=lambda x: x[0])
    top = rs[:10]
    if not top:
        return 0.0
    rel = sum(1 for _, isrel in top if isrel)
    return rel / 10.0

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_scripts/score_eval.py <path/to/labels.tsv>")
        sys.exit(1)
    labels_path = Path(sys.argv[1])
    if not labels_path.exists():
        raise SystemExit(f"[score] Not found: {labels_path}")

    by_query = defaultdict(list)
    ad_irrelevant_count = defaultdict(int)

    for row in read_rows(labels_path):
        q = row["query"]
        by_query[q].append(row)
        if not as_bool(row.get("is_relevant")) and as_bool(row.get("is_ad")):
            ad_irrelevant_count[q] += 1

    per_query = []
    for q, rows in by_query.items():
        per_query.append({
            "query": q,
            "p@10": round(p_at_10(rows), 3),
            "ads_irrelevant": ad_irrelevant_count[q]
        })

    macro = 0.0 if not per_query else sum(x["p@10"] for x in per_query) / len(per_query)
    out = {
        "macro_p@10": round(macro, 3),
        "n_queries": len(per_query),
        "per_query": per_query,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
