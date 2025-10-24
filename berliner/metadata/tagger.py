
# Phase 5 — Metadata Tagging / Structuring
# Unifies chunks + summaries into a single enriched JSONL:
#   data/enriched/metadata.jsonl
#
# Notes:
# - Robust issue_id parsing with fallbacks for inconsistent filenames.
# - Adds synthetic 'title' (from summary/chunk) and a short 'preview'.
# - Passes through page info if available upstream (page_span/pages/page_range).
# - Keeps schema + stats + manifest for reproducibility.
# ============================================================

from __future__ import annotations

import json
import re
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------
# Small helpers
# -----------------------------
def sha1(s: str) -> str:
    return "sha1:" + hashlib.sha1(s.encode("utf-8")).hexdigest()

def word_count(txt: str) -> int:
    return len((txt or "").split())

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _first_words(s: str, n: int) -> str:
    return " ".join((s or "").strip().split()[:n])

def _synth_title(summary_text: str | None, chunk_text: str | None) -> str | None:
    """
    Create a human-friendly title when no explicit title exists.
    Prefer the summary; fallback to the first words of the chunk.
    """
    if summary_text and summary_text.strip():
        return _first_words(summary_text, 12)
    if chunk_text and chunk_text.strip():
        return _first_words(chunk_text, 12)
    return None

# -----------------------------
# Issue-id parsing
# -----------------------------
def parse_issue_id(name: str, pat: str) -> tuple[str, int | None, int | None]:
    """
    Try several patterns:
    - Configured YYYY[-_]MM (preferred)
    - Any YYYY ... MM within 0-3 chars
    - Issue number only (e.g., ..._230_...) → returns TB_230 with year/month=None
    """
    # 1) config pattern (YYYY[-_]MM)
    m = re.search(pat, name)
    if m:
        y, mth = int(m.group(1)), int(m.group(2))
        return f"TB_{y}_{mth:02d}", y, mth

    # 2) loose year-month anywhere
    m = re.search(r"(20\d{2}).{0,3}([01]\d)", name)
    if m:
        y, mth = int(m.group(1)), int(m.group(2))
        return f"TB_{y}_{mth:02d}", y, mth

    # 3) issue number only (accepts end-of-string after digits)
    m = re.search(r"[_-](\d{2,4})(?:[^0-9].*|$)", name)
    if m:
        num = int(m.group(1))
        return f"TB_{num}", None, None

    # Give up (caller will fallback to a sanitized stem)
    raise ValueError(f"Cannot parse issue_id from: {name}")

# -----------------------------
# Main builder
# -----------------------------
def build_metadata(cfg: dict) -> dict:
    t0 = time.time()
    meta = cfg["metadata"]
    paths = meta["paths"]
    pdf_dir = Path(paths["pdf_dir"])
    chunks_dir = Path(paths["chunks_dir"])
    sums_dir = Path(paths["summaries_dir"])
    out_dir = Path(paths["enriched_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "metadata.jsonl"
    out_schema = out_dir / "meta_schema.json"
    out_stats  = out_dir / "stats.json"
    manifest   = out_dir / "_manifest.csv"

    pat = meta["issue_id_regex"]
    default_lang = meta["default_lang"]
    parser_ver = meta["parser_version"]

    # Map PDFs by derived issue_id (skip unparseable filenames)
    pdf_map: dict[str, tuple[str, int, int, int]] = {}
    for p in pdf_dir.glob("*.pdf"):
        try:
            iid, y, mth = parse_issue_id(p.name, pat)
            pdf_map[iid] = (str(p), p.stat().st_size, y, mth)
        except ValueError:
            # Inconsistent naming? skip; pipeline can proceed without this mapping
            continue

    counts = dict(
        issues=0, chunks=0, summaries=0, written=0,
        missing_pdf=0, missing_summary=0
    )
    seen_issue_ids: set[str] = set()
    manifest_lines = ["issue_id,issue_date,chunk_file,summary_file,num_records"]

    with out_jsonl.open("w", encoding="utf-8") as w:
        for chunk_file in sorted(chunks_dir.glob("*.jsonl")):
            stem = chunk_file.stem
            sum_file = sums_dir / f"{stem}.jsonl"
            if not sum_file.exists():
                counts["missing_summary"] += 1
                continue

            # Derive issue_id with robust fallbacks
            try:
                issue_id, year, month = parse_issue_id(stem, pat)
            except Exception:
                # Try reading first record to find a hint
                try:
                    first = next(iter_jsonl(chunk_file))
                    fallback = first.get("issue_file", first.get("issue_id", stem))
                except StopIteration:
                    fallback = stem
                try:
                    issue_id, year, month = parse_issue_id(str(fallback), pat)
                except Exception:
                    # FINAL FALLBACK: sanitize stem → stable issue_id with unknown date
                    norm = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
                    issue_id, year, month = f"TB_{norm}", None, None

            pdf_info = pdf_map.get(issue_id)
            if not pdf_info:
                counts["missing_pdf"] += 1
                pdf_path, size = None, None
                y_map = m_map = None
            else:
                pdf_path, size, y_map, m_map = pdf_info
                # prefer explicitly parsed year/month; else use mapped ones
                year = year if isinstance(year, int) else y_map
                month = month if isinstance(month, int) else m_map

            # Guard unknown year/month
            issue_date = f"{year}-{month:02d}" if (isinstance(year, int) and isinstance(month, int)) else None
            if issue_id not in seen_issue_ids:
                counts["issues"] += 1
                seen_issue_ids.add(issue_id)

            n = 0
            for idx, (chunk, summ) in enumerate(zip(iter_jsonl(chunk_file), iter_jsonl(sum_file))):
                n += 1
                counts["chunks"] += 1
                counts["summaries"] += 1

                # Upstream field names (be defensive)
                c_txt = (
                    chunk.get("chunk_text")
                    or chunk.get("text")
                    or chunk.get("body")
                    or ""
                )
                s_txt = (
                    summ.get("summary")
                    or summ.get("text")
                    or summ.get("summary_text")
                    or ""
                )

                # Construct unified record
                rec = {
                    "issue_id": issue_id,
                    "issue_date": issue_date,
                    "year": year,
                    "month": month,

                    "source_pdf_path": pdf_path,
                    "parser_version": parser_ver,

                    "chunk_id": f"{issue_id}#c_{idx:04d}",
                    "chunk_index": idx,
                    "chunk_word_count": word_count(c_txt),
                    "chunk_text_hash": sha1(c_txt) if c_txt else None,
                    "chunk_text": c_txt,  # NEW: include chunk text

                    "summary_text": s_txt,
                    "summary_word_count": word_count(s_txt),
                    "summarizer_model": summ.get("model", meta["summarizer_defaults"]["model"]),
                    "summarizer_params": summ.get(
                        "params",
                        {"num_beams": meta["summarizer_defaults"]["num_beams"]}
                    ),
                    "summary_text_hash": sha1(s_txt) if s_txt else None,

                    "runtime_s_extract": chunk.get("_meta", {}).get("runtime_s"),
                    "runtime_s_summarize": summ.get("_meta", {}).get("runtime_s"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "lang": chunk.get("lang", default_lang),
                    "file_size_bytes": size,

                    # ------- NEW FIELDS FOR SEARCH/UI -------
                    # Try to pass-through any page info if present upstream; else None
                    "page_span": (
                        chunk.get("page_span")
                        or chunk.get("pages")
                        or chunk.get("page_range")
                        or None
                    ),
                    # Prefer explicit titles if they exist; else synthesize from summary/chunk
                    "title": (
                        chunk.get("title")
                        or chunk.get("headline")
                        or chunk.get("section_title")
                        or _synth_title(s_txt, c_txt)
                    ),
                    # Short, human-friendly preview line for validation & UI
                    "preview": (
                        (s_txt.strip() if s_txt and s_txt.strip() else _first_words(c_txt, 30))
                    ),
                }

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["written"] += 1

            manifest_lines.append(f"{issue_id},{issue_date},{chunk_file.name},{sum_file.name},{n}")

    # Schema
    schema = {
        "version": "0.1.0",
        "primary_key": ["chunk_id"],
        "fields": {
            "issue_id": "str",
            "issue_date": "YYYY-MM?",
            "year": "int?",
            "month": "int?",
            "source_pdf_path": "str?",
            "parser_version": "str",

            "chunk_id": "str",
            "chunk_index": "int",
            "chunk_word_count": "int",
            "chunk_text_hash": "str?",
            "chunk_text": "str?",          # NEW

            "summary_text": "str",
            "summary_word_count": "int",
            "summarizer_model": "str",
            "summarizer_params": "object",
            "summary_text_hash": "str?",

            "runtime_s_extract": "float?",
            "runtime_s_summarize": "float?",
            "created_at": "datetime",
            "lang": "str",
            "file_size_bytes": "int?",

            # NEW fields for search/UI
            "page_span": "str?",
            "title": "str?",
            "preview": "str?",
        }
    }

    # Write schema/stats/manifest
    out_schema.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    counts["seconds_total"] = round(time.time() - t0, 2)
    out_stats.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return counts
