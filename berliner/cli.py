# berliner/cli.py
from __future__ import annotations

import sys
import hashlib
from pathlib import Path
import click

# our modules
from berliner.utils.config import load_config
from berliner.extractor.parse import extract_issue, save_json

# ---- CLI root ----
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
    """The Berliner CLI — extraction, chunking, and (soon) embeddings & search."""
    cfg = load_config(config_path)
    ctx.obj = {"cfg": cfg}

# ---- extract command ----
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
    """Extract one PDF or a folder of PDFs into JSON issue files."""
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
            h = hashlib.sha1(pdf.name.encode()).hexdigest()[:8]
            stem = f"{stem}.{h}"
        out_path = save_json(data, str(out_dir), stem=stem)
        click.echo(f"[extract] Saved {out_path}")

# ---- chunk command (Phase 3) ----
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
    """Chunk a single issue JSON file or all JSON files in a folder."""
    cfg = ctx.obj["cfg"]
    out_dir = out_dir or Path(cfg["paths"]["chunks"])
    size_words = size_words or int(cfg["chunking"]["size_words"])
    overlap_words = overlap_words or int(cfg["chunking"]["overlap_words"])

    # Import the concrete implementations
    try:
        from berliner.chunker.make_chunks import chunk_issue_file, chunk_dir
    except Exception as e:
        click.echo(f"[chunk] Import error: {e}", err=True)
        sys.exit(1)

    # discover inputs
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

# ---- entrypoint ----
def main():
    cli(prog_name="berliner")

if __name__ == "__main__":
    main()
