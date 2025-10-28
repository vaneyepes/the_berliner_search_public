# Stage 7 — Minimal UI: Search tab only

from __future__ import annotations
import json, pathlib
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

# ----  import existing search function exactly as CLI uses ----
_import_error = None
dense_query = None
try:
    from berliner.search.indexer import query as dense_query  # <- do not change backend
except Exception as e:
    _import_error = e

# ---- Small helpers ----
def path_exists(p: str | pathlib.Path) -> bool:
    return pathlib.Path(p).exists()

def normalize_hits(hits: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
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

# ---- Page config & light brand ----
st.set_page_config(page_title="The Berliner — Archive Search", page_icon="🔎", layout="wide")

st.markdown("""
<style>
:root { --accent: #E62619; }
.stButton>button[kind="primary"] { background: var(--accent); color:#fff; border:1px solid var(--accent); border-radius:8px; }
.result-card { border:1px solid #e9e9e9; border-left:4px solid var(--accent); border-radius:8px; padding:0.9rem 1rem; margin-bottom:0.75rem; background:#fff; }
.score-badge { display:inline-block; padding:2px 8px; border:1px solid var(--accent); border-radius:999px; font-size:0.8rem; color:var(--accent); }
.soft-sep { border-top:1px solid #eee; margin:0.75rem 0; }
.brand-underline { height:3px; background: var(--accent); margin: 0.25rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar (logo + settings) ----
st.sidebar.header("Settings")
logo_path = st.sidebar.text_input("Logo path", value="assets/TheBerliner_Logo_1000px_RGB.png")
if path_exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

# ---- Header ----
st.markdown("<h1>The Berliner — Archive Search</h1>", unsafe_allow_html=True)
st.markdown('<div class="brand-underline"></div>', unsafe_allow_html=True)
if _import_error:
    st.error(
        "Could not import `berliner.search.indexer.query`.\n\n"
        "Make sure you ran `pip install -e .` in this venv.\n\n"
        f"Details: {_import_error}"
    )

# ---- Search UI ----
st.markdown("### Enter your query")
q = st.text_input(
    "Query",
    placeholder="e.g., Berlin airport delays, gentrification in Neukölln",
    label_visibility="collapsed",
)
cols = st.columns([0.2, 0.8])
with cols[0]:
    submit = st.button("Search", type="primary", use_container_width=True)
with cols[1]:
    st.caption("Dense retrieval (MiniLM). Top-10 results.")

st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)

if submit and q and dense_query is not None:
    try:
        # Keep it simple: depend on backend defaults (model, faiss_top, etc.)
        hits = dense_query(text=q, k=10)  # signature from your backend
    except TypeError:
        # In case your backend signature differs, try a minimal call
        try:
            hits = dense_query(q, 10)
        except Exception as e:
            st.error(f"Search failed: {e}")
            hits = []
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
                  <div style="color:#555; font-size:0.9rem; margin-top:0.4rem;">
                    {"Summary available" if has_summary else "No summary"} · Dense retrieval
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No results. Try another query.")
