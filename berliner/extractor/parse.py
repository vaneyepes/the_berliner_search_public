from typing import Dict, Any
from pathlib import Path
import json

from .engines import (
    extract_blocks_pymupdf,
    cluster_columns,
    strip_header_footer,
    order_blocks,
)
from .utils import dehyphenate, infer_meta_from_name, now_utc_iso

# Use a single, exact token (no surrounding newlines). New extracts will use this.
PAGE_BREAK = "<<<PAGE_BREAK>>>"

def extract_issue(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts text blocks from a PDF, removes headers/footers, orders columns,
    joins pages with PAGE_BREAK, applies light cleaning, and returns a dict:
      { "meta": {...}, "text": "..." }
    """
    pdf_path = Path(pdf_path)

    # ---- 1) Extract blocks and clean layout ----
    blocks = extract_blocks_pymupdf(str(pdf_path))
    blocks = strip_header_footer(blocks, header_pct=0.08, footer_pct=0.08)
    blocks = cluster_columns(blocks, k_candidates=(1, 2, 3))
    blocks = order_blocks(blocks)

    # ---- 2) Join page texts with the exact PAGE_BREAK token ----
    texts = []
    current_page = None
    for b in blocks:
        if current_page is None:
            current_page = b["page"]
        if b["page"] != current_page:
            texts.append(PAGE_BREAK)  # exact token, no extra newlines
            current_page = b["page"]
        # Keep each block tidy; paragraph spacing handled below
        txt = (b.get("text") or "").strip()
        if txt:
            texts.append(txt)

    # Preserve paragraph spacing but avoid runaway whitespace
    joined = "\n\n".join([t for t in texts if t])
    cleaned = dehyphenate(joined)

    # ---- 3) Meta: infer + standardize + defaults for downstream stages ----
    meta = infer_meta_from_name(pdf_path.name) or {}
    n_pages = (blocks[-1]["page"] + 1) if blocks else 0

    # Stable ID carried across all stages (matches file stem)
    meta.setdefault("issue_id", pdf_path.stem)         # e.g., TheBerliner243_2025_08_21
    # Optional date if infer_meta_from_name extracted it; else None is fine
    meta.setdefault("date", meta.get("date"))          # e.g., "2025-08-21" or None
    # Provenance + reproducibility
    meta.setdefault("n_pages", n_pages)
    meta.setdefault("parse_engine", "pymupdf")
    meta.setdefault("clean_version", "v1")
    meta.setdefault("page_break_token", PAGE_BREAK)
    meta.setdefault("source_file", str(pdf_path))
    meta.setdefault("extracted_at", now_utc_iso())

    return {"meta": meta, "text": cleaned}


def save_json(obj: Dict[str, Any], out_dir: str, stem: str | None = None) -> str:
    """
    Save a Python dict to <out_dir>/<stem>.json (UTF-8, pretty).
    Returns the filesystem path as string.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = (stem or "issue") + ".json"
    path = out / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)
