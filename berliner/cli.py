# berliner/cli.py
from __future__ import annotations 

import sys
import hashlib
from pathlib import Path
import click

# Import functions from berliner modules
from berliner.utils.config import load_config
from berliner.extractor.parse import extract_issue, save_json

# from berliner.chunker.make_chunks import chunk_issue_file, chunk_dir
from berliner.summarizer.summarize import summarize_issue

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

# ---- summarize command (Phase 4) ----
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
    """
    Summarize a single chunk JSONL file or all JSONL files in a folder.

    INPUT:
      - input_path: either a directory containing *.jsonl chunks, or one .jsonl file.

    OUTPUT:
      - One JSONL per input file in the summaries directory, with records:
        {chunk_id, summary, model, gen:{...}} and a final _meta line.
    """
    cfg = ctx.obj["cfg"]

    # Pull summarization defaults from config.yaml
    try:
        s = cfg["summarization"]
    except KeyError:
        click.echo("[summarize] Missing 'summarization' section in config.yaml", err=True)
        sys.exit(1)

    # Resolve paths and params (overrides > config)
    chunks_dir_default = Path(cfg["paths"]["chunks"]) if "paths" in cfg and "chunks" in cfg["paths"] else None
    out_dir = out_dir or Path(s.get("output_dir", "data/summaries"))
    model = model_name or s.get("model_name", "t5-small")
    max_input_tokens = int(max_input_tokens or s.get("max_input_tokens", 512))
    max_new_tokens  = int(max_new_tokens  or s.get("max_new_tokens", 120))
    num_beams       = int(num_beams       or s.get("num_beams", 4))
    batch_size      = int(batch_size      or s.get("batch_size", 4))
    device          = device or s.get("device", "auto")
    skip_if_exists  = not force  # invert the --force flag

    # Discover inputs: directory of *.jsonl vs single file
    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.jsonl"))
        # If a directory was passed but empty, try falling back to configured chunks dir
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

    # Run per file
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
                skip_if_exists=skip_if_exists,  # supported by our summarize.py
            )
            if out is None:
                click.echo(f"[summarize] Skipped (exists) {f.stem}")
            else:
                click.echo(f"[summarize] Wrote {out}")
                n_written += 1
        except Exception as e:
            click.echo(f"[summarize] ERROR on {f.name}: {e}", err=True)

    click.echo(f"[summarize] Done. New/updated files: {n_written}")


# ---- entrypoint ----
def main():
    cli(prog_name="berliner")

if __name__ == "__main__":
    main()


