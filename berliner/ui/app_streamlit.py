from __future__ import annotations

import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import quote as _urlquote

import streamlit as st

# ========= Config =========
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOGO = "assets/TheBerliner_Logo.svg"
ACCENT = "#E62619"

# Data locations in repo
DEFAULT_ENRICHED_META = pathlib.Path("data/enriched/metadata.jsonl")
DEFAULT_CHUNKS_DIR    = pathlib.Path("data/chunks")
DEFAULT_SUMMARIES_DIR = pathlib.Path("data/summaries")
RAW_PDF_DIR           = pathlib.Path("data/raw_pdfs")

def _autodetect_ids_path() -> str:
    base = pathlib.Path("data/index")
    if not base.exists():
        return ""
    candidates = list(base.rglob("ids.jsonl"))
    if not candidates:
        return ""
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)

INDEX_IDS_PATH = _autodetect_ids_path()

# ========= Small helpers =========
def _file_uri(p: pathlib.Path | str) -> str:
    try:
        p = pathlib.Path(p).resolve()
        return "file://" + _urlquote(str(p))
    except Exception:
        return ""

def _val_ok(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    return True

# ========= CLI Search =========
@st.cache_data(show_spinner=False)
def run_cli_search(query: str, k: int = 10, model: str = DEFAULT_MODEL) -> List[Tuple[float, Dict[str, Any]]]:
    cmd = [sys.executable, "-m", "berliner.cli", "search", query, "--model", model, "-k", str(int(k))]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    results: List[Tuple[float, Dict[str, Any]]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line or "score=" not in line or "chunk=" not in line:
            continue
        parts = shlex.split(line)
        kv = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip().strip(",")
        try:
            score = float(kv.get("score", "0"))
        except Exception:
            score = 0.0
        rec: Dict[str, Any] = {
            "issue_id": kv.get("issue") or kv.get("issue_id"),
            "chunk_id": kv.get("chunk") or kv.get("chunk_id"),
            "title": kv.get("title", ""),
        }
        rec["summary_text"] = ""
        rec["chunk_text"] = ""
        rec["has_summary"] = False
        results.append((score, rec))
    return results

# ========= Loaders =========
@st.cache_data(show_spinner=False)
def load_enriched_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    p = pathlib.Path(meta_path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            cid = rec.get("chunk_id")
            if not cid:
                continue
            out[cid] = {
                "summary_text": (rec.get("summary_text") or "").strip(),
                "chunk_text":   (rec.get("chunk_text") or rec.get("preview") or "").strip(),
                "title":        (rec.get("title") or rec.get("section_title") or rec.get("headline") or "").strip(),
                "page_span": rec.get("page_span"),
                "issue_id":  rec.get("issue_id"),
                "source_pdf_path": rec.get("source_pdf_path"),
            }
    return out

@st.cache_data(show_spinner=False)
def load_lookup_from_chunks(chunks_dir: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    d = pathlib.Path(chunks_dir)
    if not d.exists():
        return lookup
    for fp in d.glob("*.jsonl"):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    cid = obj.get("chunk_id") or obj.get("id")
                    if not cid:
                        continue
                    rec = lookup.setdefault(cid, {})
                    if "text" in obj and obj["text"]:
                        rec.setdefault("chunk_text", obj["text"][:600])
                    if "title" in obj and obj["title"]:
                        rec.setdefault("title", obj["title"])
                    if "page_span" in obj and obj["page_span"]:
                        rec.setdefault("page_span", obj["page_span"])
                    if "issue_id" in obj and obj["issue_id"]:
                        rec.setdefault("issue_id", obj["issue_id"])
        except Exception:
            continue
    return lookup

@st.cache_data(show_spinner=False)
def load_lookup_from_summaries(summaries_dir: str) -> Dict[str, Dict[str, Any]]:
    look: Dict[str, Dict[str, Any]] = {}
    d = pathlib.Path(summaries_dir)
    if not d.exists():
        return look
    for fp in d.glob("*.jsonl"):
        if fp.name.startswith("_catalog"):
            continue
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    cid = obj.get("chunk_id") or obj.get("id")
                    if not cid:
                        continue
                    rec = look.setdefault(cid, {})
                    if "summary" in obj and obj["summary"]:
                        rec.setdefault("summary_text", obj["summary"])
                    if "page_span" in obj and obj["page_span"]:
                        rec.setdefault("page_span", obj["page_span"])
                    if "issue_id" in obj and obj["issue_id"]:
                        rec.setdefault("issue_id", obj["issue_id"])
        except Exception:
            continue
    return look

@st.cache_data(show_spinner=False)
def load_ids_map(ids_path: str) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    path = pathlib.Path(ids_path)
    if not path or not path.exists():
        return m
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            cid = obj.get("chunk_id") or obj.get("id") or obj.get("chunk")
            if not cid:
                meta = obj.get("meta") or {}
                cid = meta.get("chunk_id") or meta.get("id") or meta.get("chunk")
            if not cid:
                continue
            rec: Dict[str, Any] = {}
            if isinstance(obj.get("title"), str) and obj["title"]:
                rec["title"] = obj["title"]
            head = obj.get("chunk_text") or obj.get("head") or obj.get("text") or ""
            if isinstance(head, str) and head:
                rec["chunk_text"] = head[:600]
            if obj.get("page_span"):
                rec["page_span"] = obj["page_span"]
            if obj.get("issue_id"):
                rec["issue_id"] = obj["issue_id"]
            if isinstance(obj.get("summary_text"), str) and obj["summary_text"]:
                rec["summary_text"] = obj["summary_text"]
            m[cid] = rec
    return m

# ========= Merge Logic (non-destructive) =========
def _safe_merge(base: Dict[str, Any], add: Dict[str, Any], keys: List[str]):
    for k in keys:
        if not _val_ok(base.get(k)) and _val_ok(add.get(k)):
            base[k] = add[k]

def enrich_hits_merged(
    hits: List[Tuple[float, Dict[str, Any]]],
    enriched_map: Dict[str, Dict[str, Any]],
    chunks_map: Dict[str, Dict[str, Any]],
    sums_map: Dict[str, Dict[str, Any]],
    ids_map: Dict[str, Dict[str, Any]],
) -> List[Tuple[float, Dict[str, Any]]]:
    out: List[Tuple[float, Dict[str, Any]]] = []
    for score, rec in hits:
        cid = rec.get("chunk_id", "")
        merged = dict(rec)
        fields = {
            "issue_id": merged.get("issue_id"),
            "title": merged.get("title"),
            "page_span": merged.get("page_span"),
            "summary_text": merged.get("summary_text"),
            "chunk_text": merged.get("chunk_text"),
            "source_pdf_path": merged.get("source_pdf_path"),
        }
        for source in (enriched_map, chunks_map, sums_map, ids_map):
            if cid in source:
                _safe_merge(fields, source[cid],
                            ["issue_id", "title", "page_span", "summary_text", "chunk_text", "source_pdf_path"])
        if not _val_ok(fields["issue_id"]) and "#" in cid:
            fields["issue_id"] = cid.split("#", 1)[0]

        if not _val_ok(fields.get("source_pdf_path")) and _val_ok(fields.get("issue_id")):
            iss = str(fields["issue_id"])
            cand1 = RAW_PDF_DIR / f"{iss}.pdf"
            if cand1.exists():
                fields["source_pdf_path"] = str(cand1)
            else:
                parts = iss.split("_")
                ym = None
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    ym = (parts[1], parts[2])
                candidates = []
                if ym:
                    for pat in (f"*{ym[0]}_{ym[1]}*.pdf", f"*{ym[0]}-{ym[1]}*.pdf"):
                        candidates.extend(RAW_PDF_DIR.glob(pat))
                if not candidates:
                    candidates = list(RAW_PDF_DIR.glob(f"*{iss}*.pdf"))
                if candidates:
                    best = max(candidates, key=lambda p: p.stat().st_mtime)
                    fields["source_pdf_path"] = str(best)

        merged.update(fields)
        merged["has_summary"] = bool(merged.get("summary_text"))
        out.append((score, merged))
    return out

# -------- Preview/truncation logic --------
def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut if cut else text[:n]) + "…"

def rows_from_hits(
    hits: List[Tuple[float, Dict[str, Any]]],
    prefer_summary: bool = True
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for score, rec in hits:
        summary = (rec.get("summary_text") or "").strip()
        chunk_head = (rec.get("chunk_text") or "").strip()

        if prefer_summary and summary:
            preview = _truncate(summary, 800)  # longer summaries
        else:
            preview = ""  # hide body when summaries are off

        rows.append({
            "score": float(score),
            "issue": rec.get("issue_id"),
            "chunk": rec.get("chunk_id"),
            "pages": rec.get("page_span"),
            "title": (rec.get("title") or "").strip(),
            "preview": preview,
            "has_summary": bool(summary),
            "pdf_path": rec.get("source_pdf_path"),
        })
    return rows
# ----------------------------------------------------

# ========= UI =========
st.set_page_config(page_title="The Berliner — Smart Archive Search", page_icon="🔎", layout="wide")

st.markdown(f"""
<style>
:root {{ --accent: {ACCENT}; }}
.result-card {{border:1px solid #e9e9e9;border-left:4px solid var(--accent);border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem;background:#fff;}}
.score-badge {{display:inline-block;padding:2px 8px;border:1px solid var(--accent);border-radius:999px;font-size:0.8rem;color:var(--accent);}}
.soft-sep {{ border-top:1px solid #eee; margin:0.75rem 0; }}
.brand-underline {{ height:3px; background: var(--accent); margin: 0.25rem 0 1rem 0; }}

/* Logo container styles */
.logo-left {{ margin: 0.5rem 0 0.35rem 0; display:flex; justify-content:flex-start; }}
.logo-left .logo-wrap {{ max-width: 360px; }}
.logo-left svg {{ width: 100%; height: auto; display: block; }}
</style>
""", unsafe_allow_html=True)

# Sidebar — Search Settings
st.sidebar.header("Search Settings")
results_limit = st.sidebar.slider("Number of results to show", 3, 25, 10)
show_summaries = st.sidebar.toggle("Show article summaries", True)
include_similar = st.sidebar.toggle("Include similar topics", True)  # (placeholder)
sort_by = st.sidebar.selectbox("Sort results by", ["Relevance", "Chronological"])

# --- Logo (left-aligned, SVG) ---
logo_path = pathlib.Path(DEFAULT_LOGO)
if logo_path.exists():
    try:
        svg_text = logo_path.read_text(encoding="utf-8")
        st.markdown(f"<div class='logo-left'><div class='logo-wrap'>{svg_text}</div></div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div class='logo-left'><div class='logo-wrap'>", unsafe_allow_html=True)
        st.image(str(logo_path), use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

# Header
st.markdown("<h1>Smart Archive Search</h1>", unsafe_allow_html=True)
st.markdown('<div class="brand-underline"></div>', unsafe_allow_html=True)
st.caption("AI-assisted search engine for The Berliner ePaper repository • Powered by semantic search")

# Tabs
tab_search, tab_dash = st.tabs(["🔎 Search", "📊 Dashboard"])

with tab_search:
    with st.form(key="search-form"):
        q = st.text_input(
            "Query",
            placeholder="e.g., Berlin airport delays, gentrification in Neukölln, citizenship applications",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and q:
        t0 = time.time()
        try:
            with st.spinner("Searching…"):
                raw_hits = run_cli_search(q, k=int(results_limit), model=DEFAULT_MODEL)

                enriched_map = load_enriched_meta(str(DEFAULT_ENRICHED_META))
                chunks_map   = load_lookup_from_chunks(str(DEFAULT_CHUNKS_DIR))
                sums_map     = load_lookup_from_summaries(str(DEFAULT_SUMMARIES_DIR))
                ids_map      = load_ids_map(INDEX_IDS_PATH)

                hits = enrich_hits_merged(raw_hits, enriched_map, chunks_map, sums_map, ids_map)
                rows = rows_from_hits(hits, prefer_summary=show_summaries)

            # --- Sort results if user selected "Chronological" ---
            def _ym_from_issue(issue: str | None) -> tuple[int, int]:
                if not issue or "_" not in issue:
                    return (0, 0)
                parts = str(issue).split("_")
                nums = [p for p in parts if p.isdigit()]
                if len(nums) >= 2:
                    y, m = nums[0], nums[1]
                    try:
                        return (int(y), int(m))
                    except Exception:
                        return (0, 0)
                return (0, 0)

            if sort_by == "Chronological":
                rows.sort(key=lambda r: _ym_from_issue(r.get("issue")), reverse=True)
            # -------------------------------------------------------

        except subprocess.CalledProcessError as e:
            st.error(f"CLI error:\n\n{e.output}")
            rows = []
        except Exception as e:
            st.error(f"Search failed: {e}")
            rows = []

        elapsed = time.time() - t0
        st.caption(f"Top {results_limit} · Took {elapsed:.2f}s")

        if rows:
            st.markdown("### Results")
            for r in rows:
                issue = r.get("issue") or "—"
                chunk = r.get("chunk") or "—"
                pages = r.get("pages", None)
                title = r.get("title", "")
                preview = r.get("preview", "")
                score = r.get("score", 0.0)
                has_summary = r.get("has_summary", False)
                pdf_path = r.get("pdf_path")

                pages_str = f" · pages {pages[0]}–{pages[1]}" if isinstance(pages, list) and len(pages) == 2 else ""
                title_str = f" — *{title}*" if title else ""

                pdf_html = ""
                if pdf_path and pathlib.Path(pdf_path).exists():
                    pdf_html = f'<div style="margin-top:0.25rem;"><a href="{_file_uri(pdf_path)}" target="_blank" style="font-weight:600; color: var(--accent);">Open PDF of this edition</a></div>'

                status_text = (
                    ("Summary available" if has_summary else "No summary")
                    if show_summaries
                    else ("Summary hidden" if has_summary else "No summary")
                )
                # Displasys summaries if enabled, else shows chunk preview only if no summary
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div><span class="score-badge">Relevance {score*100:.0f}%</span></div>
                      <div style="margin-top: 0.35rem;">
                        <strong>{issue}</strong> · <code>{chunk}</code>{pages_str}{title_str}
                      </div>
                      <div class="soft-sep"></div>
                      <div>{preview}</div>
                      <div style="color:#555; font-size:0.9rem; margin-top:0.4rem;">
                        {status_text} · Dense retrieval (CLI)
                      </div>
                      {pdf_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No results. Try another query.")

with tab_dash:
    st.info("Dashboard coming next. Reads existing stats/listings only (no new processing).")
