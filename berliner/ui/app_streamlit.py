# Initial UI for The Berliner Search
# ------------------------------------------------------------
# - Provides a clean Search tab that calls your existing search function:
#       from berliner.search.indexer import query as dense_query
# - Displays ranked results with issue, pages, score, and snippet
# - Provides a light "Dashboard" tab that reads existing JSON/JSONL
#
# How to run (from repo root, in venv):
#   streamlit run berliner/ui/app_streamlit.py
# ------------------------------------------------------------

from __future__ import annotations
import json
import pathlib
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd  # used for dashboard summaries (lightweight)

# ---------- BRAND / APP CONFIG ----------
ACCENT = "#E62619"          #  brand red
DEFAULT_TOP_K = 10            # top-k results default
DEFAULT_FAISS_TOP = 80        # rerank pool for FAISS candidates
DEFAULT_HYBRID = False        # hybrid off by default (dense-only)

# Default paths (can override these from the sidebar at runtime)
DEFAULT_PATHS = {
    "logo": "assets/TheBerliner_Logo_1000px_RGB.png",
    "faiss_stats_minilm": "data/index/sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2/stats.json",
    "faiss_ids_minilm": "data/index/sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2/ids.jsonl",
    "chunks_dir": "data/chunks",
    "summaries_dir": "data/summaries",
    "enriched_metadata": "data/enriched/metadata.jsonl",
}


# ---------- IMPORT SEARCH ENTRYPOINT ----------
# This is the function existing in berliner/search/indexer.py:
# results = dense_query(text: str, k: int = 10, hybrid: bool = True, faiss_top: int = 80)
try:
    from berliner.search.indexer import query
    dense_query = query  # alias for clarity
    _import_error = None
except Exception as e:
    dense_query = None
    _import_error = e

# ---------- LIGHT STYLING ----------
def inject_brand_styles(accent_hex: str, use_webfonts: bool = True) -> None:
    """
    Inject minimal CSS to:
    - Load League Gothic (titles) + Merriweather (body) if available
    - Set accent color
    - Style result cards
    """
    webfont_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=League+Gothic:wdth,wght@100..125,400&family=Merriweather:wght@300;400;700&display=swap');
    </style>
    """ if use_webfonts else ""

    css = f"""
    {webfont_css}
    <style>
      :root {{ --accent: {accent_hex}; }}

      /* Body font defaults */
      .stApp, .stMarkdown, .stText, .stDataFrame, p, li {{
        font-family: {'Merriweather, Georgia, Times New Roman, serif' if use_webfonts else 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif'};
        font-size: 16px;
      }}

      /* Titles: League Gothic if loaded, otherwise fallback */
      h1, h2, h3, .section-title {{
        font-family: {'League Gothic, Impact, Anton, sans-serif' if use_webfonts else 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif'};
        letter-spacing: 0.2px;
      }}

      /* Accented buttons */
      .stButton>button, .stDownloadButton>button {{
        border-radius: 8px;
        border: 1px solid var(--accent);
      }}
      .stButton>button[kind="primary"], .stDownloadButton>button {{
        background-color: var(--accent);
        color: #fff;
      }}

      .brand-underline {{
        height: 3px;
        background: var(--accent);
        margin: 0.25rem 0 1rem 0;
      }}

      .result-card {{
        border: 1px solid #e9e9e9;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        background: #fff;
      }}

      .meta {{
        color: #555;
        font-size: 0.9rem;
      }}

      .score-badge {{
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid var(--accent);
        border-radius: 999px;
        font-size: 0.8rem;
        color: var(--accent);
      }}

      .soft-sep {{
        border-top: 1px solid #eee;
        margin: 0.75rem 0;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------- FILE HELPERS ----------
def path_exists(p: str | pathlib.Path) -> bool:
    return pathlib.Path(p).exists()

def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load small JSON files (e.g., stats.json). Returns None if missing/invalid."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def iter_jsonl(path: str | pathlib.Path):
    """Yield dicts from a JSONL file; skip bad lines silently."""
    p = pathlib.Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ---------- NORMALIZE SEARCH RESULTS ----------
def normalize_hits(hits: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert [(score, rec), ...] into a uniform dict we can render.
    Pull: issue_id, chunk_id, page_span, title, summary_text/chunk_text head.
    """
    rows: List[Dict[str, Any]] = []
    for score, rec in hits:
        issue = rec.get("issue_id") or rec.get("issue")
        chunk = rec.get("chunk_id") or rec.get("chunk")
        pages = rec.get("page_span") or rec.get("pages")
        title = (rec.get("title") or "").strip()
        summary = (rec.get("summary_text") or "").strip()
        chunk_head = (rec.get("chunk_text") or "").strip()
        preview = summary[:300] if summary else chunk_head[:300]
        rows.append({
            "score": float(score),
            "issue": issue,
            "chunk": chunk,
            "pages": pages,
            "title": title,
            "preview": preview or "",
            "has_summary": bool(rec.get("has_summary", bool(summary))),
        })
    return rows


# ---------- APP START ----------
st.set_page_config(
    page_title="The Berliner — Archive Search",
    page_icon="🔎",
    layout="wide",
)
inject_brand_styles(ACCENT, use_webfonts=True)

# ---------- SIDEBAR ----------
st.sidebar.header("Settings")
logo_path = st.sidebar.text_input("Logo path", value=DEFAULT_PATHS["logo"])
accent = st.sidebar.color_picker("Accent color", value=ACCENT)
top_k = st.sidebar.slider("Top-K results", 3, 30, DEFAULT_TOP_K, 1)
faiss_top = st.sidebar.slider("FAISS candidates (pool)", 40, 320, DEFAULT_FAISS_TOP, 20)
hybrid = st.sidebar.toggle("Hybrid search (BM25 + dense if available)", value=DEFAULT_HYBRID)

with st.sidebar.expander("Dashboard data (optional)", expanded=False):
    fp_stats = st.text_input("FAISS stats (MiniLM)", value=DEFAULT_PATHS["faiss_stats_minilm"])
    fp_ids = st.text_input("FAISS ids (MiniLM)", value=DEFAULT_PATHS["faiss_ids_minilm"])
    dir_chunks = st.text_input("Chunks dir", value=DEFAULT_PATHS["chunks_dir"])
    dir_summ = st.text_input("Summaries dir", value=DEFAULT_PATHS["summaries_dir"])
    fp_meta = st.text_input("Enriched metadata", value=DEFAULT_PATHS["enriched_metadata"])

# ---------- HEADER ----------
cols = st.columns([0.18, 0.82])
with cols[0]:
    if path_exists(logo_path):
        st.image(logo_path, use_container_width=True)
with cols[1]:
    st.markdown("<h1>Smart Archive Search</h1>", unsafe_allow_html=True)
    st.markdown('<div class="brand-underline"></div>', unsafe_allow_html=True)
    st.caption("Semantic search over The Berliner’s PDF archive")

# ---------- TABS ----------
tab_search, tab_dashboard = st.tabs(["🔎 Search", "📊 Dashboard"])

# --- SEARCH TAB ---
with tab_search:
    if _import_error:
        st.error(
            "Could not import `berliner.search.indexer.query`. "
            "Make sure you installed the package in editable mode and activated the venv.\n\n"
            f"Details: {_import_error}"
        )

    st.markdown("### Enter your query")
    q = st.text_input(
        "Query",
        placeholder="e.g., Berlin airport delays, gentrification in Neukölln, citizenship applications",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([0.2, 0.8])
    with c1:
        submit = st.button("Search", type="primary", use_container_width=True)
    with c2:
        st.caption(f"Top **{top_k}** • FAISS pool {faiss_top} • Hybrid = {hybrid}")

    st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)

    if submit and q and dense_query is not None:
        try:
            # Call your existing search implementation
            hits = dense_query(text=q, k=int(top_k), hybrid=bool(hybrid), faiss_top=int(faiss_top))
        except Exception as e:
            st.error(f"Search failed: {e}")
            hits = []

        rows = normalize_hits(hits) if hits else []

        if rows:
            st.markdown("#### Results")
            for r in rows:
                issue = r.get("issue", "")
                chunk = r.get("chunk", "")
                pages = r.get("pages", None)
                title = r.get("title", "")
                preview = r.get("preview", "")
                score = r.get("score", 0.0)
                has_summary = r.get("has_summary", False)

                # Pretty formatting for pages if [start, end]
                pages_str = f" · pages {pages[0]}–{pages[1]}" if isinstance(pages, list) and len(pages) == 2 else ""
                title_str = f" — *{title}*" if title else ""

                st.markdown(
                    f"""
                    <div class="result-card">
                      <div><span class="score-badge">score {score:.4f}</span></div>
                      <div style="margin-top: 0.35rem;">
                        <strong>{issue}</strong> · <code>{chunk}</code>{pages_str}{title_str}
                      </div>
                      <div class="soft-sep"></div>
                      <div>{preview if preview else "<em>No preview available</em>"}</div>
                      <div class="meta" style="margin-top: 0.4rem;">
                        {"Summary available" if has_summary else "No summary"} · Dense retrieval
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        elif submit:
            st.info("No results. Try a different query or increase FAISS candidates.")

# --- DASHBOARD TAB (MVP-light, reads existing files only) ---
with tab_dashboard:
    st.markdown("### Archive Dashboard (MVP-light)")
    st.caption("Reads existing JSON/JSONL files at runtime. No new files will be generated.")

    # KPI placeholders
    kcol = st.columns(5)
    total_issues = "—"
    total_chunks = "—"
    summary_coverage = "—"
    vec_dim = "—"
    vec_count = "—"

    # From FAISS stats.json (MiniLM)
    stats = load_json(fp_stats)
    if stats:
        vec_dim = stats.get("embedding_dim") or stats.get("dim") or "—"
        vec_count = stats.get("num_vectors") or stats.get("count") or "—"

    # From ids.jsonl — counts & issue set
    ids_count = 0
    issues_set = set()
    if path_exists(fp_ids):
        for rec in iter_jsonl(fp_ids):
            ids_count += 1
            issue = rec.get("issue_id")
            if not issue:
                chunk_id = rec.get("chunk_id", "")
                if "#" in chunk_id:
                    issue = chunk_id.split("#", 1)[0]
            if issue:
                issues_set.add(issue)

    if issues_set:
        total_issues = len(issues_set)
    if ids_count:
        total_chunks = ids_count

    # Summary coverage (rough count across summaries/*.jsonl)
    summary_count = 0
    if path_exists(dir_summ):
        for f in pathlib.Path(dir_summ).glob("*.jsonl"):
            if f.name.startswith("_catalog"):
                continue
            for _ in iter_jsonl(str(f)):
                summary_count += 1
    if ids_count:
        summary_coverage = f"{(summary_count / ids_count * 100):.1f}%" if summary_count else "0.0%"

    with kcol[0]:
        st.metric("Issues indexed", total_issues)
    with kcol[1]:
        st.metric("Chunks (vectors)", total_chunks)
    with kcol[2]:
        st.metric("Summaries coverage", summary_coverage)
    with kcol[3]:
        st.metric("Vector dim", vec_dim)
    with kcol[4]:
        st.metric("Total vectors", vec_count)

    st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)
    st.info(
        "Next: add charts (chunks per issue, coverage by issue, issues over time) "
        "using your existing JSONL. This tab currently avoids heavy parsing."
    )

# ---------- FOOTER ----------
st.caption("© The Berliner — Internal Smart Search tool by Vanesa Yepes")
