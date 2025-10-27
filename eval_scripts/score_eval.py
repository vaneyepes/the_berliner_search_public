from pathlib import Path
import csv, json, statistics

base = Path("eval")
runs = sorted(base.rglob("labels.tsv"))
assert runs, "No labels.tsv found. Did you save it next to results.jsonl?"
labels_path = runs[-1]
run_dir = labels_path.parent
jsonl_path = run_dir / "results.jsonl"

rows = []
with open(labels_path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        r["is_relevant"] = (r["is_relevant"] or "").strip().upper().startswith("Y")
        r["is_ad"] = (r["is_ad"] or "").strip().upper().startswith("Y")
        rows.append(r)

by_q = {}
for r in rows:
    by_q.setdefault(r["query"], []).append(r)

def p_at_10(items):
    items = sorted(items, key=lambda x: int(x["rank"]))[:10]
    return sum(1 for i in items if i["is_relevant"]) / 10.0

def ad_at_10(items):
    items = sorted(items, key=lambda x: int(x["rank"]))[:10]
    return sum(1 for i in items if i["is_ad"]) / 10.0

per_query = []
for q, items in by_q.items():
    per_query.append({
        "query": q,
        "p@10": round(p_at_10(items), 3),
        "ad@10": round(ad_at_10(items), 3),
    })

macro = {
    "macro_p@10": round(statistics.mean([x["p@10"] for x in per_query]), 3),
    "macro_ad@10": round(statistics.mean([x["ad@10"] for x in per_query]), 3),
}

print("Per-query:")
for x in per_query: print(x)
print("\nMacro:", macro)

with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump({"per_query": per_query, **macro}, f, ensure_ascii=False, indent=2)

print(f"✅ Wrote {run_dir/'summary.json'}")
