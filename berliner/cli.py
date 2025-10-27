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

# ---- Earlier phase imports ----
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
# Phase 1–5 commands (unchanged, shortened for brevity)
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
# Stage 6 — Embeddings + FAISS (new)
# ==========================================================
@cli.command("embed-index")
@click.argument("json_dir", type=click.Path(exists=True, path_type=Path)
