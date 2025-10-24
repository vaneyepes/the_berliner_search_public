# ============================
# The Berliner Search — Unified CLI
# Stages covered:
#   Phase 1–2: Extraction (PDF → JSON issues)
#   Phase 3:   Chunking (issue JSON → chunked JSONL)
#   Phase 4:   Summarization (chunks → summaries JSONL)
#   Phase 5:   Metadata Tagging (chunks + summaries → enriched metadata.jsonl)
#   Phase 6:   Semantic Search (embeddings + FAISS, hybrid BM25, optional rerank)
# ============================

from __future__ import annotations

import sys
from pathlib import Path
import click
import re

# ---- Common config loader ----
# Loads config.yaml (project-wide paths & params)
from berliner.utils.config import load_config

# ---- Phase 1–2: Extraction imports ----
# PDF -> structured issue JSON
from berliner.extractor.parse import extract_issue, save_json

# ---- Phase 4: Summarization import ----
# Chunks -> summaries JSONL
from berliner.summarizer.summarize import summarize_issue

# ---- Phase 5: Metadata Tagging import ----
# Chunks + summaries -> enriched metadata.jsonl (+ schema, stats, manifest)
from berliner.metadata.tagger import build_metadata


# -----------------------------------------------------------------------------
# CLI root
# -----------------------------------------------------------------------------
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (defaults to project-root config.yaml).",
    show_default=False,
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None):
    """
    The Berliner CLI — extraction, chunking, summarization, metadata tagging, and search.
    """
    cfg = load_config(config_path)
    ctx.obj = {"cfg": cfg}


# -----------------------------------------------------------------------------
# Phase 1–2: extract
# PDF(s) → issue JSON files in data/json/
# Why: normalize raw PDFs into machine-readable issues for later stages.
# -----------------------------------------------------------------------------
@cli.command("extract")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--out", "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for issue JSON files (defaults to paths.issues_json from config).",
)
@click.option(
    "--hash-stem/--no-hash-stem",
    default=False,
    show_default=True,
    help="Append an 8-char hash to the filename stem to avoid collisions.",
)
@click.pass_context
def extract_cmd(ctx: click.Context, input_path: Path, out_dir: Path | None, hash_stem: bool):
    cfg = ctx.obj["cfg"]
    out_dir = out_dir or Path(cfg["paths"]["issues_json"])

    if input_path.is_dir():
        pdf_files = sorted(p for p in input_path.glob("*.pdf"))
        if not pdf_files:
            click.echo(f"[extract] No PDFs found in {input_path}", err=True)
            sys.exit(1)
        click.echo(f"[extract] Found {len(pdf_files)} PDFs in {input_path}")
    else:
        pdf_files = [input_path]

    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_files:
        data = extract_issue(str(pdf))
        stem = pdf.stem
        if hash_stem:
            import hashlib
            h = hashlib.sha1(pdf.name.encode().decode() if isinstance(pdf.name, bytes) else pdf.name.encode()).hexdigest()[:8]
            stem = f"{stem}.{h}"
        out_path = save_json(data, str(out_dir), stem=stem)
        click.echo(f"[extract] Saved {out_path}")


# -----------------------------------------------------------------------------
# Phase 3: chunk
# Issue JSON → chunked JSONL in data/chunks/
# Why: split long articles into ~N-word windows for downstream NLP.
# -----------------------------------------------------------------------------
@cli.command("chunk")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--out", "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for chunk JSONL files (defaults to paths.chunks from config).",
)
@click.option(
    "--size", "size_words",
    type=int, default=None,
    help="Approximate words per chunk (defaults to chunking.size_words in config).",
)
@click.option(
    "--overlap", "overlap_words",
    type=int, default=None,
    help="Words of overlap between consecutive chunks (defaults to chunking.overlap_words in config).",
)
@click.pass_context
def chunk_cmd(
    ctx: click.Context,
    input_path: Path,
    out_dir: Path | None,
    size_words: int | None,
    overlap_words: int | None,
):
    cfg = ctx.obj["cfg"]
    out_dir = out_dir or Path(cfg["paths"]["chunks"])
    size_words = size_words or int(cfg["chunking"]["size_words"])
    overlap_words = overlap_words or int(cfg["chunking"]["overlap_words"])

    # Import inside to avoid hard dependency when user only runs other stages
    try:
        from berliner.chunker.make_chunks import chunk_issue_file, chunk_dir
    except Exception as e:
        click.echo(f"[chunk] Import error: {e}", err=True)
        sys.exit(1)

    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.json"))
        if not files:
            click.echo(f"[chunk] No .json files in {input_path}", err=True)
            sys.exit(1)
        click.echo(f"[chunk] Found {len(files)} issue JSON files in {input_path}")
        out_path_list = chunk_dir(
            input_path, out_dir,
            chunk_words=size_words, overlap_words=overlap_words
        )
        for p in out_path_list:
            click.echo(f"[chunk] Wrote {p}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = chunk_issue_file(
            input_path, out_dir,
            chunk_words=size_words, overlap_words=overlap_words
        )
        click.echo(f"[chunk] Wrote {p}")


# -----------------------------------------------------------------------------
# Phase 4: summarize
# Chunked JSONL → summaries JSONL in data/summaries/
# Why: produce concise, model-generated summaries for each chunk.
# -----------------------------------------------------------------------------
@cli.command("summarize")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--out", "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for summaries JSONL files (defaults to summarization.output_dir in config).",
)
@click.option("--model-name", default=None, help='Override model (defaults to summarization.model_name, e.g., "t5-small" or "facebook/bart-base").')
@click.option("--max-input-tokens", type=int, default=None, help="Override summarization.max_input_tokens.")
@click.option("--max-new-tokens", type=int, default=None, help="Override summarization.max_new_tokens.")
@click.option("--num-beams", type=int, default=None, help="Override summarization.num_beams.")
@click.option("--batch-size", type=int, default=None, help="Override summarization.batch_size.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default=None, help='Override summarization.device ("auto" by default).')
@click.option("--limit", type=int, default=0, help="Optional: limit number of files processed (useful for quick tests).")
@click.option("--force/--no-force", default=False, show_default=True, help="Force re-write even if output exists (disables idempotent skip).")
@click.pass_context
def summarize_cmd(
    ctx: click.Context,
    input_path: Path,
    out_dir: Path | None,
    model_name: str | None,
    max_input_tokens: int | None,
    max_new_tokens: int | None,
    num_beams: int | None,
    batch_size: int | None,
    device: str | None,
    limit: int,
    force: bool,
):
    cfg = ctx.obj["cfg"]

    try:
        s = cfg["summarization"]
    except KeyError:
        click.echo("[summarize] Missing 'summarization' section in config.yaml", err=True)
        sys.exit(1)

    chunks_dir_default = Path(cfg["paths"]["chunks"]) if "paths" in cfg and "chunks" in cfg["paths"] else None
    out_dir = out_dir or Path(s.get("output_dir", "data/summaries"))
    model = model_name or s.get("model_name", "t5-small")
    max_input_tokens = int(max_input_tokens or s.get("max_input_tokens", 512))
    max_new_tokens  = int(max_new_tokens  or s.get("max_new_tokens", 120))
    num_beams       = int(num_beams       or s.get("num_beams", 4))
    batch_size      = int(batch_size      or s.get("batch_size", 4))
    device          = device or s.get("device", "auto")
    skip_if_exists  = not force

    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.jsonl"))
        if not files and chunks_dir_default and input_path == chunks_dir_default:
            click.echo(f"[summarize] No .jsonl files in {input_path}", err=True)
            sys.exit(1)
        if limit and limit > 0:
            files = files[:limit]
        click.echo(f"[summarize] Found {len(files)} chunk files in {input_path}")
    else:
        if input_path.suffix.lower() != ".jsonl":
            click.echo(f"[summarize] Expected a .jsonl file, got: {input_path.name}", err=True)
            sys.exit(1)
        files = [input_path]

    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(
        f"[summarize] model={model} device={device} beams={num_beams} "
        f"max_in={max_input_tokens} max_new={max_new_tokens} batch={batch_size} "
        f"→ out={out_dir}"
    )

    n_written = 0
    for f in files:
        try:
            out = summarize_issue(
                issue_path=f,
                out_dir=out_dir,
                model_name=model,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                batch_size=batch_size,
                device=device,
                skip_if_exists=skip_if_exists,
            )
            if out is None:
                click.echo(f"[summarize] Skipped (exists) {f.stem}")
            else:
                click.echo(f"[summarize] Wrote {out}")
                n_written += 1
        except Exception as e:
            click.echo(f"[summarize] ERROR on {f.name}: {e}", err=True)

    click.echo(f"[summarize] Done. New/updated files: {n_written}")


# -----------------------------------------------------------------------------
# Phase 5: metadata
# Chunks + summaries → data/enriched/{metadata.jsonl, meta_schema.json, stats.json, _manifest.csv}
# Why: unify/normalize fields for embeddings & search (Phase 6).
# -----------------------------------------------------------------------------
@cli.command("metadata")
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.pass_context
def metadata_cmd(ctx: click.Context, config: str):
    cfg = ctx.obj["cfg"]  # loaded by CLI root
    # If user passed a different --config here, reload with that path:
    if config and config != ctx.parent.params.get("config_path"):
        cfg = load_config(config)

    counts = build_metadata(cfg)
    click.echo(
        f"[metadata] wrote {counts['written']} rows in {counts['seconds_total']}s "
        f"(issues={counts.get('issues', 0)}, chunks={counts.get('chunks', 0)}, "
        f"missing_summary={counts.get('missing_summary', 0)})"
    )


# -----------------------------------------------------------------------------
# Phase 6: search
# Embeddings + FAISS index (and querying) in data/index/
# Why: enable semantic retrieval using Sentence-Transformers + FAISS.
# -----------------------------------------------------------------------------
@cli.group("search")
@click.pass_context
def search_group(ctx: click.Context):
    """Semantic search utilities (Phase 6)."""
    # No-op: group holder; reads config via parent ctx when needed.
    pass


@search_group.command("index")
@click.option("--index-type",
              type=click.Choice(["flat", "hnsw"]),
              default="flat",
              show_default=True,
              help="FAISS index type: exact (flat) or ANN graph (hnsw).")
@click.option("--batch-size", type=int, default=None, show_default=False,
              help="Embedding batch size (overrides search.batch_size in config).")
@click.option("--no-normalize", is_flag=True, help="Disable L2 normalization (not recommended).")
@click.option("--model-name", default=None, show_default=False,
              help="Override embedding model (defaults to search.model_name or MiniLM).")
@click.option("--max-length", type=int, default=None, show_default=False,
              help="Override search.max_length (tokens).")
@click.pass_context
def search_index_cmd(
    ctx: click.Context,
    index_type: str,
    batch_size: int | None,
    no_normalize: bool,
    model_name: str | None,
    max_length: int | None,
):
    """
    Build embeddings and a FAISS index.

    Outputs:
      - data/index/embeddings.npy
      - data/index/ids.jsonl
      - data/index/faiss.index
      - data/index/stats.json (augmented with FAISS info)
    """
    cfg = ctx.obj["cfg"]
    cfg_search = cfg.get("search", {}) if isinstance(cfg, dict) else {}

    # Resolve runtime parameters with config fallbacks
    model_name = model_name or cfg_search.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(batch_size or cfg_search.get("batch_size", 64))
    normalize = not no_normalize if "normalize" not in cfg_search else bool(cfg_search.get("normalize", True) and not no_normalize)
    max_length = int(max_length or cfg_search.get("max_length", 512))

    # Import inside command to keep optional dependency surface small
    try:
        from berliner.search.embedder import run as build_embeddings
        from berliner.search.indexer import build_index
    except Exception as e:
        click.echo(f"[search] Import error: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"[search] embeddings → model={model_name} batch={batch_size} "
        f"normalize={normalize} max_length={max_length}"
    )
    try:
        stats = build_embeddings(
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size,
            max_length=max_length,
        )
        click.echo(f"[search] embeddings: dim={stats.get('dim')} n={stats.get('n_vectors')} time={stats.get('elapsed_sec')}s")
    except Exception as e:
        click.echo(f"[search] ERROR building embeddings: {e}", err=True)
        sys.exit(1)

    click.echo(f"[search] building FAISS index: index_type={index_type}")
    try:
        info = build_index(index_type=index_type)
        click.echo(f"[search] index: type={info.get('index_type')} n={info.get('n_vectors')} dim={info.get('dim')} time={info.get('elapsed_sec')}s")
    except Exception as e:
        click.echo(f"[search] ERROR building FAISS index: {e}", err=True)
        sys.exit(1)


@search_group.command("query")
@click.argument("text", type=str)
@click.option("-k", "--topk", type=int, default=None, show_default=False,
              help="Top-K results to display (defaults to search.k in config or 10).")
@click.option("--show-meta", is_flag=True, help="Print the raw metadata mapping for each hit.")
@click.option("--no-hybrid", is_flag=True, help="Disable BM25+dense fusion; use dense-only FAISS retrieval.")
@click.option("--faiss-top", type=int, default=80, show_default=True,
              help="FAISS candidate pool size before fusion/rerank (ignored if --no-hybrid).")
@click.option("--rerank", is_flag=True,
              help="Apply cross-encoder reranking (ms-marco-MiniLM-L-6-v2) after fusion for higher precision.")
@click.pass_context
def search_query_cmd(ctx, text, topk, show_meta, no_hybrid, faiss_top, rerank):
    """
    Run a semantic query against the FAISS index.

    Prints: rank, score, issue_id, chunk_id, page_span, (optional) title.
    """
    cfg = ctx.obj["cfg"]
    cfg_search = cfg.get("search", {}) if isinstance(cfg, dict) else {}
    k = int(topk or cfg_search.get("k", 10))

    try:
        from berliner.search.indexer import query as faiss_query
    except Exception as e:
        click.echo(f"[search] Import error: {e}", err=True)
        sys.exit(1)

    # 1) Dense (or Hybrid) retrieval
    try:
        results = faiss_query(
            text,
            k=k if no_hybrid else max(k, faiss_top),
            hybrid=not no_hybrid,
            faiss_top=faiss_top,
        )
    except AssertionError as e:
        click.echo(f"[search] AssertionError: {e}", err=True)
        click.echo("[search] Tip: build the index first:  python -m berliner.cli search index", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[search] ERROR running query: {e}", err=True)
        sys.exit(1)

    # 2) Optional reranking (cross-encoder) — use summary + chunk head for better precision
    if rerank and results:
        try:
            from sentence_transformers import CrossEncoder

            def _chunk_head_local(s: str, n_words: int = 120) -> str:
                return " ".join((s or "").strip().split()[:n_words])

            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = []
            for _, r in results:
                # Prefer summary + chunk head; fallback to preview/title if empty
                txt = (r.get("summary_text") or "") + "\n\n" + _chunk_head_local(r.get("chunk_text") or "", 120)
                if not txt.strip():
                    txt = r.get("preview") or r.get("title") or ""
                pairs.append((text, txt))

            ce_scores = ce.predict(pairs).tolist()
            results = sorted(zip(ce_scores, [r for _, r in results]), key=lambda x: x[0], reverse=True)[:k]
        except Exception as e:
            click.echo(f"[search] Rerank error: {e}", err=True)

    if not results:
        click.echo("[search] No results.")
        return

    # 3) Print
    for rank, (score, rec) in enumerate(results, start=1):
        issue = rec.get("issue_id"); chunk = rec.get("chunk_id")
        pages = rec.get("page_span"); title = rec.get("title") or ""
        click.echo(f"{rank:02d}  score={score:.4f}  issue={issue}  chunk={chunk}  pages={pages}  {title}")
        if show_meta:
            import json as _json
            click.echo("     " + _json.dumps(rec, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    cli(prog_name="berliner")

if __name__ == "__main__":
    main()
