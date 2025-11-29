from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

# ----------------------------------------------------
#  Demo-only Streamlit UI for the public case-study
#  This version does NOT call the internal CLI or
#  access any real data. It only shows synthetic
#  example results so that the layout and interaction
#  pattern are visible without touching private data.
# ----------------------------------------------------

ACCENT = "#7C3AED"

# Completely synthetic, non-realistic example results
DEMO_RESULTS: List[Dict[str, Any]] = [
    {
        "score": 0.89,
        "issue_id": "TB_2024_01",
        "title": "Example article about Berlin airport delays",
        "summary_text": "Placeholder summary text used only for this public demo UI.",
        "page_span": [4, 5],
    },
    {
        "score": 0.83,
        "issue_id": "TB_2023_06",
        "title": "Example feature on housing and gentrification",
        "summary_text": "Second placeholder summary. No real magazine content is shown here.",
        "page_span": [10, 12],
    },
    {
        "score": 0.78,
        "issue_id": "TB_2022_11",
        "title": "Example interview with a local artist",
        "summary_text": "Third placeholder summary for UI demonstration purposes.",
        "page_span": [2, 3],
    },
]


def render_result_card(rec: Dict[str, Any]) -> None:
    score = rec.get("score", 0.0)
    issue_id = rec.get("issue_id") or "Unknown issue"
    title = rec.get("title") or "Untitled article"
    summary = rec.get("summary_text") or "No summary available."
    page_span = rec.get("page_span") or []

    pages = ""
    if isinstance(page_span, (list, tuple)) and page_span:
        if len(page_span) == 1:
            pages = f"p. {page_span[0]}"
        else:
            pages = f"pp. {page_span[0]}–{page_span[-1]}"

    st.markdown(
        f"""
<div class="result-card">
  <div class="result-header">
    <span class="score">{score:.2f}</span>
    <span class="issue">{issue_id}</span>
    <span class="pages">{pages}</span>
  </div>
  <div class="result-title">{title}</div>
  <div class="result-summary">{summary}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="The Berliner — Smart Archive Search (Demo)",
        page_icon="🔎",
        layout="wide",
    )

    st.markdown(
        f"""
<style>
:root {{
  --accent: {ACCENT};
}}
.result-card {{
  border: 1px solid #e9e9e9;
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.75rem;
  background-color: #ffffff;
}}
.result-header {{
  display: flex;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: #6b7280;
  margin-bottom: 0.2rem;
}}
.result-header .score {{
  font-weight: 600;
  color: var(--accent);
}}
.result-title {{
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 0.2rem;
}}
.result-summary {{
  font-size: 0.9rem;
  color: #374151;
}}
.info-pill {{
  display: inline-flex;
  align-items: center;
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  background-color: #f9fafb;
  border: 1px solid #e5e7eb;
  font-size: 0.78rem;
  color: #4b5563;
  margin-right: 0.4rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("🔎 The Berliner Search — Demo UI (Public Case Study)")

    st.markdown(
        """
This Streamlit app is a **demo-only UI** for the public case-study of *The Berliner Search*.

- No real magazine content is loaded.
- No PDFs, metadata, embeddings, or FAISS index files are accessed.
- The example results below are fully synthetic and exist only to illustrate the layout and interaction flow.

For a private walkthrough of the full internal system (with real data), please get in touch.
"""
    )

    with st.sidebar:
        st.subheader("Demo Settings")
        st.markdown(
            """
This sidebar mimics some of the controls from the internal tool.

In the production version, you can:

- Choose the embedding model
- Adjust *k* (number of results)
- Switch between different evaluation presets

In this public demo, the controls are disabled to avoid suggesting that a real search is running.
"""
        )
        _ = st.text_input("Query (disabled in demo)", value="Berlin airport delays", disabled=True)
        _ = st.slider("Top-k results", 1, 20, 10, disabled=True)
        st.caption("Controls disabled in demo mode.")

    st.markdown("### Example results (synthetic)")

    for rec in DEMO_RESULTS:
        render_result_card(rec)

    st.info(
        "This is a **static demo** using placeholder data only. "
        "In the internal version, results are computed via FAISS-based semantic search "
        "over the magazine's archive."
    )


if __name__ == "__main__":
    main()
