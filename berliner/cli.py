# ============================
# The Berliner Search — Unified CLI
# Stages covered:
#   Phase 1–2: Extraction (PDF → JSON issues)
#   Phase 3:   Chunking (issue JSON → chunked JSONL)
#   Phase 4:   Summarization (chunks → summaries JSONL)
#   Phase 5:   Metadata Tagging (chunks + summaries → enriched metadata.jsonl)
#   Phase 6:   Semantic Search (embeddings + FAISS)
# ============================

from __future__ import annotations
import sys
from pathlib import Path
import click
import json

from berliner.utils.config import load_config
from berliner.extractor.parse import extract_issue, save_json
from berliner.summarizer.summarize import summarize_issue
from berliner.metadata.tagger import build_metadata

# ==========================================================
# CLI root
# ==========================================================
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (defaults to project root).",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None):
    """The Berliner CLI — extraction, chunking, summarization, metadata tagging, and search."""
    cfg = load_config(config_path)
    ctx.obj = {"cfg": cfg}

# ==========================================================
# Phase 1–5 commands (kept simple)
# ==========================================================
@cli.command("extract")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", "out_dir", type=click.Path(path_type=Path))
@click.pass_context
def extract_cmd(ctx, input_path, out_dir):
    cfg = ctx.obj["cfg"]
    out_dir = out_dir or Path(cfg["paths"]["issues_json"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
    for pdf in pdfs:
        data = extract_issue(str(pdf))
        out_path = save_json(data, str(out_dir), stem=pdf.stem)
        click.echo(f"[extract] Saved {out_path}")

@cli.command("summarize")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", "out_dir", type=click.Path(path_type=Path))
@click.pass_context
def summarize_cmd(ctx, input_path, out_dir):
    cfg = ctx.obj["cfg"]
    s = cfg["summarization"]
    out_dir = out_dir or Path(s["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summarize_issue(issue_path=input_path, out_dir=out_dir, model_name=s["model_name"])
    click.echo(f"[summarize] Wrote summaries to {out_dir}")

@cli.command("metadata")
@click.pass_context
def metadata_cmd(ctx):
    cfg = ctx.obj["cfg"]
    counts = build_metadata(cfg)
    click.echo(
        f"[metadata] wrote {counts['written']} rows "
        f"(issues={counts.get('issues',0)}, chunks={counts.get('chunks',0)})"
    )

# ==========================================================
# Stage 6 — Embeddings + FAISS
# ==========================================================
@cli.command("embed-index")
@click.argument("json_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    required=True,
    help='HF model name (e.g. "sentence-transformers/multi-qa-mpnet-base-dot-v1")'
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    help="Batch size for encoding."
)
@click.pass_context
def embed_index_cmd(ctx, json_dir, model, batch_size):
    """
    Build embeddings + FAISS index for a specific model.
    Saves outputs under:
      data/embeddings/<model-slug>/
      data/index/<model-slug>/
    """
    import os
    from berliner.search.indexer import build_index

    click.echo("[debug] embed-index: command started")
    click.echo(f"[debug] cwd={os.getcwd()}")
    click.echo(f"[debug] json_dir arg={json_dir}")

    # ---- Loader that supports BOTH metadata.jsonl and per-issue *.json with 'chunks'
    def load_chunks(dir_path: Path):
        chunks = []

        # 1) metadata.jsonl (enriched)
        meta_path = dir_path / "metadata.jsonl"
        if meta_path.exists():
            click.echo(f"[debug] using {meta_path} (metadata.jsonl)")
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = (rec.get("summary_text") or rec.get("chunk_text") or "").strip()
                cid  = rec.get("chunk_id")
                if text and cid:
                    chunks.append({"chunk_id": cid, "text": text})

        # 2) fallback: per-issue JSON files with {'chunks': [{chunk_id,text},...]}
        if not chunks:
            json_files = sorted(dir_path.glob("*.json*"))
            if json_files:
                click.echo(f"[debug] scanning {len(json_files)} issue JSON files in {dir_path}")
            for p in json_files:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for c in data.get("chunks", []):
                    t = (c.get("text") or "").strip()
                    cid = c.get("chunk_id")
                    if t and cid:
                        chunks.append({"chunk_id": cid, "text": t})

        return chunks

    dir_path = Path(json_dir)
    chunks = load_chunks(dir_path)
    click.echo(f"[debug] chunks loaded: {len(chunks)}")

    if not chunks:
        click.echo(f"[embed-index] No chunks found in {json_dir}", err=True)
        click.echo("[embed-index] Tip: use 'data/enriched' (metadata.jsonl) or a folder of issue JSONs with a 'chunks' array.", err=True)
        import sys as _sys
        _sys.exit(1)

    cfg = ctx.obj["cfg"]
    emb_root = Path(cfg["paths"]["embeddings"])
    idx_root = Path(cfg["paths"]["index"])
    click.echo(f"[debug] embeddings_root={emb_root}")
    click.echo(f"[debug] index_root={idx_root}")

    info = build_index(
        chunks=chunks,
        model_name=model,
        embeddings_root=emb_root,
        index_root=idx_root,
        batch_size=batch_size,
    )
    click.echo(json.dumps(info, indent=2))
    click.echo("[embed-index] done.")

# ==========================================================
# Stage 6 — Minimal search command (with snippets for eval)
# ==========================================================
@cli.command("search")
@click.argument("query", type=str)
@click.option("--model", required=True, help="Model used to build the index (same HF id).")
@click.option("-k", default=10, show_default=True, help="Top-K results to show.")
def search_cmd(query, model, k):
    """
    Simple FAISS similarity search using a chosen model’s index.
    Prints a short snippet for each hit so you can label relevance in eval.
    Example:
      python -m berliner.cli search "Berlin airport delays" \
        --model "sentence-transformers/multi-qa-mpnet-base-dot-v1" -k 10
    """
    from pathlib import Path
    import json as _json
    import faiss
    from berliner.search.embedder import Embedder

    model_slug = model.replace("/", "__")
    idx_dir = Path("data/index") / model_slug
    idx_path = idx_dir / "faiss.index"
    ids_path = idx_dir / "ids.jsonl"

    if not idx_path.exists():
        raise SystemExit(f"[search] Index not found: {idx_path}. Build it with 'embed-index' first.")

    index = faiss.read_index(str(idx_path))
    ids = [_json.loads(line)["chunk_id"] for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    embedder = Embedder(model)
    q = embedder.encode([query])  # normalized
    scores, idxs = index.search(q, k)
    scores, idxs = scores[0].tolist(), idxs[0].tolist()

    # ---- build a preview map: chunk_id -> short snippet (summary_text or chunk_text) ----
    meta_path = Path("data/enriched/metadata.jsonl")
    chunk_preview = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = _json.loads(line)
            cid = rec.get("chunk_id")
            txt = (rec.get("summary_text") or rec.get("chunk_text") or "").strip()
            if cid and txt and cid not in chunk_preview:
                words = txt.split()
                snippet = " ".join(words[:80]) + (" ..." if len(words) > 80 else "")
                chunk_preview[cid] = snippet

    # ---- print ranked results with a readable snippet for labeling ----
    click.echo(f"\nTop {k} results for: {query}\n")
    for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
        cid = ids[i] if 0 <= i < len(ids) else "<missing>"
        click.echo(f"{rank:02d}\tscore={s:.4f}\tchunk={cid}")
        snip = chunk_preview.get(cid, "")
        if snip:
            click.echo(f"     {snip}")

# -----------------------------------------------------------------------------
# Entrypoint (robust + diagnostic)
# -----------------------------------------------------------------------------
def main():
    import click as _click, traceback
    try:
        _click.echo("[debug] cli: main() entered")
        cli(prog_name="berliner")
    except SystemExit as e:
        _click.echo(f"[debug] cli: SystemExit({e.code})", err=False)
        raise
    except Exception:
        import sys as _sys
        _click.echo("[debug] cli: unhandled exception during startup:", err=True)
        traceback.print_exc()
        _sys.exit(1)

if __name__ == "__main__":
    main()
