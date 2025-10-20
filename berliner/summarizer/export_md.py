from pathlib import Path
import json

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_issue_md(summ_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    issue_id = summ_file.stem
    chunks, meta = [], None
    for rec in iter_jsonl(summ_file):
        if "_meta" in rec: meta = rec["_meta"]; continue
        chunks.append(rec)
    # quick 1–2 sentence issue summary from first few chunks
    top = " ".join([c["summary"] for c in chunks[:4]])
    if len(top) > 900: top = top[:900] + "…"

    md = [f"# {issue_id}", "", f"**Auto issue summary:** {top}", "", "## Chunks"]
    for c in chunks:
        span = c.get("page_span", [])
        span_str = f"pp. {span[0]}–{span[1]}" if span else ""
        md.append(f"- `{c['chunk_id']}` {span_str}: {c['summary']}")
    if meta:
        md += ["", f"_model: {meta.get('model')} · device: {meta.get('device')} · chunks: {meta.get('n_chunks')} · runtime: {meta.get('runtime_s')}s_"]
    out_path = out_dir / f"{issue_id}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path
