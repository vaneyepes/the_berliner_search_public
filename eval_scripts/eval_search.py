import subprocess, shlex, re, json, time
from pathlib import Path

# ---- config (edit TOPK/FAISS_TOP/RERANK only if needed) ----
QUERIES = [q.strip() for q in Path("queries.txt").read_text(encoding="utf-8").splitlines() if q.strip()]
TOPK = 10
FAISS_TOP = 160
RERANK = True
RUN_TAG = f"{'rerank' if RERANK else 'baseline'}_k{TOPK}_faiss{FAISS_TOP}"
OUT_DIR = Path("eval") / time.strftime("%Y%m%d_%H%M%S") / RUN_TAG
# ------------------------------------------------------------

def run_query(q: str) -> str:
    cmd = f'python -m berliner.cli search query {shlex.quote(q)} -k {TOPK} --faiss-top {FAISS_TOP} '
    if RERANK:
        cmd += "--rerank"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def parse_output(txt: str):
    pat = r"^\s*(\d+)\s+score=([-\d\.]+)\s+issue=([^\s]+)\s+chunk=([^\s]+)\s+pages=\[[^\]]*\]\s+(.*)$"
    rows = []
    for line in txt.splitlines():
        m = re.match(pat, line)
        if m:
            rows.append({
                "rank": int(m.group(1)),
                "score": float(m.group(2)),
                "issue": m.group(3),
                "chunk": m.group(4),
                "snippet": m.group(5).strip()
            })
    return rows

OUT_DIR.mkdir(parents=True, exist_ok=True)
jsonl_path = OUT_DIR / "results.jsonl"
txt_path = OUT_DIR / "raw_results.txt"
tsv_path = OUT_DIR / "labels_template.tsv"

with txt_path.open("w", encoding="utf-8") as txt_f, jsonl_path.open("w", encoding="utf-8") as jsonl_f:
    for q in QUERIES:
        txt_f.write(f"# QUERY: {q}\n")
        out = run_query(q)
        txt_f.write(out + "\n\n")
        for row in parse_output(out):
            row["query"] = q
            row["run_tag"] = RUN_TAG
            jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

with tsv_path.open("w", encoding="utf-8") as f:
    f.write("query\trank\tissue\tchunk\tsnippet\tis_relevant\tis_ad\n")
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        snippet = r["snippet"].replace("\t", " ").replace("\n", " ")
        f.write(f"{r['query']}\t{r['rank']}\t{r['issue']}\t{r['chunk']}\t{snippet}\t\t\n")

print(f"✅ Wrote: {txt_path}")
print(f"✅ Wrote: {jsonl_path}")
print(f"✅ Wrote: {tsv_path}")
print(f"📂 Folder: {OUT_DIR}")
