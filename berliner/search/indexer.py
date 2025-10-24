"""
FAISS index builder + query utils.

- Loads embeddings from data/index/embeddings.npy
- Builds FAISS index (FlatIP for exact cosine or HNSW for large-scale)
- Saves index to data/index/faiss.index
- Provides query() helper to run top-k searches (dense-only or hybrid with BM25)
"""

from __future__ import annotations
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Tuple
import unicodedata

import numpy as np

import faiss

from sentence_transformers import SentenceTransformer
import torch

# ------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # THE_BERLINER_SEARCH/
DATA = ROOT / "data"
INDEX = DATA / "index"

EMBEDDINGS_PATH = INDEX / "embeddings.npy"
IDS_PATH = INDEX / "ids.jsonl"
FAISS_PATH = INDEX / "faiss.index"
STATS_PATH = INDEX / "stats.json"    # embedder writes model_name, dim, n_vectors; we append faiss info here

# ------------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fallback; actual model read from stats.json when possible
INDEX_TYPE = "flat"    # 'flat' or 'hnsw'
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
NORMALIZE_QUERY = True

# ------------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------------
def _chunk_head(s: str, n_words: int = 120) -> str:
    return " ".join((s or "").strip().split()[:n_words])

def _fold(s: str) -> str:
    # Lowercase + remove accents/diacritics
    return "".join(
        c for c in unicodedata.normalize("NFKD", (s or "").lower())
        if not unicodedata.combining(c)
    )

def _contains_any(text: str, terms: list[str]) -> bool:
    if not text: return False
    t = _fold(text)
    return any(tok in t for tok in terms)


def load_embeddings() -> np.ndarray:
    assert EMBEDDINGS_PATH.exists(), f"Missing embeddings: {EMBEDDINGS_PATH}"
    return np.load(EMBEDDINGS_PATH)

def load_ids() -> List[Dict[str, Any]]:
    assert IDS_PATH.exists(), f"Missing ids map: {IDS_PATH}"
    return [json.loads(line) for line in IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

def save_faiss_index(index: faiss.Index) -> None:
    faiss.write_index(index, str(FAISS_PATH))

def load_stats() -> dict:
    if STATS_PATH.exists():
        try:
            return json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ------------------------------------------------------------------------------------
# Build index
# ------------------------------------------------------------------------------------
def build_index(index_type: str = INDEX_TYPE) -> Dict[str, Any]:
    t0 = time.time()
    vecs = load_embeddings()
    n, d = vecs.shape

    if index_type == "flat":
        # IndexFlatIP expects vectors normalized if you want cosine equivalence
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
    elif index_type == "hnsw":
        # HNSW with inner product; vectors should be normalized for cosine
        index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.add(vecs)
    else:
        raise ValueError("index_type must be 'flat' or 'hnsw'")

    save_faiss_index(index)

    info = dict(
        index_type=index_type,
        n_vectors=n,
        dim=d,
        elapsed_sec=round(time.time() - t0, 2),
    )

    # Append index info into stats.json (non-destructive)
    prev = load_stats()
    prev.update(dict(faiss=info))
    STATS_PATH.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    return info

# ------------------------------------------------------------------------------------
# Internal utils (norms, model resolution, fusion)
# ------------------------------------------------------------------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype("float32")

def _resolve_model_name(default_name: str = MODEL_NAME) -> str:
    """Prefer the model_name saved during indexing (stats.json)."""
    stats = load_stats()
    return stats.get("model_name", default_name)

def _load_model(device: str = None) -> SentenceTransformer:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved = _resolve_model_name(MODEL_NAME)
    return SentenceTransformer(resolved, device=device)

def _get_index_dim(index: "faiss.Index") -> int:
    return int(getattr(index, "d", 0))

def _get_model_dim(model: "SentenceTransformer") -> int:
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:
        v = model.encode(["test"], convert_to_numpy=True)
        return int(v.shape[1])

def _prefix_for_query(model_name: str) -> str:
    name = model_name.lower()
    # Instruction-tuned retrievers like E5/GTE/BGE expect a "query: " prefix
    if "e5" in name or "gte" in name or "bge" in name:
        return "query: "
    return ""

def _rrf_merge(faiss_hits, bm25_hits, k=10, rrf_k=60):
    """
    Reciprocal Rank Fusion (RRF):
    faiss_hits: List[(score, rec_dict_with__row)]
    bm25_hits:  List[(score, rec_dict_with__row)]
    Returns top-k fused list of (score, row_index).
    """
    faiss_rank = {h[1]["_row"]: r for r, h in enumerate(faiss_hits, start=1)}
    bm25_rank  = {h[1]["_row"]: r for r, h in enumerate(bm25_hits,  start=1)}
    all_rows = set(faiss_rank) | set(bm25_rank)
    fused = []
    for row in all_rows:
        r1 = faiss_rank.get(row, 10**9)
        r2 = bm25_rank.get(row,  10**9)
        score = 1.0 / (rrf_k + r1) + 1.0 / (rrf_k + r2)
        fused.append((score, row))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused[:k]

# ------------------------------------------------------------------------------------
# Query (with hybrid BM25, RRF fusion, and domain MUST filtering)
# ------------------------------------------------------------------------------------
def query(text: str, k: int = 10, hybrid: bool = True, faiss_top: int = 80) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Encodes 'text', searches FAISS, optionally fuses with BM25 keyword search, and returns (score, id_record).

    Robustness:
      - Loads the same model used at index-build time (from stats.json).
      - Verifies model embedding dimension matches index.d before searching.
      - Falls back to dense-only if BM25 modules aren't available.

    Precision:
      - BM25 runs over title + summary + first 120 words of chunk.
      - Domain MUST filters for airport/gentrification queries to suppress generic "Berlin" hits.
    """
    # --- tiny local helpers (avoid extra imports/changes elsewhere) ---
    import unicodedata, re
    def _chunk_head_local(s: str, n_words: int = 120) -> str:
        return " ".join((s or "").strip().split()[:n_words])

    def _fold(s: str) -> str:
        # lowercase + remove accents/diacritics
        return "".join(
            c for c in unicodedata.normalize("NFKD", (s or "").lower())
            if not unicodedata.combining(c)
        )

    def _contains_any(text_: str, terms: list[str]) -> bool:
        if not text_:
            return False
        t = _fold(text_)
        return any(tok in t for tok in terms)

    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}. Run: berliner search index")

    # Load index + ID map
    try:
        index = faiss.read_index(str(FAISS_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index at {FAISS_PATH}: {e}")

    id_map = load_ids()
    if not id_map:
        raise RuntimeError("ids.jsonl is empty. Rebuild the index.")

    # ----- Dense branch (FAISS) -----
    model = _load_model()
    index_dim = _get_index_dim(index)
    model_dim = _get_model_dim(model)
    if index_dim != model_dim:
        resolved_name = _resolve_model_name(MODEL_NAME)
        raise RuntimeError(
            f"Embedding dim mismatch: index.d={index_dim}, model.d={model_dim}. "
            f"The index was built with '{resolved_name}'. Rebuild the index "
            f"with the same model you're using now, or query using that model."
        )

    # Prefix for instruction-tuned models (E5/GTE/BGE)
    resolved_name = _resolve_model_name(MODEL_NAME)
    q_prefix = _prefix_for_query(resolved_name)
    qtxt = q_prefix + text

    try:
        q = model.encode([qtxt], convert_to_numpy=True, normalize_embeddings=False)
    except Exception as e:
        raise RuntimeError(f"Failed to encode query with model {resolved_name}: {e}")

    if NORMALIZE_QUERY:
        q = _l2_normalize(q)

    pool = faiss_top if hybrid else k
    try:
        d_scores, d_idxs = index.search(q.astype("float32"), pool)
    except Exception as e:
        raise RuntimeError(f"FAISS search error: {e}")

    d_scores = d_scores[0].tolist()
    d_idxs = d_idxs[0].tolist()

    faiss_hits: List[Tuple[float, Dict[str, Any]]] = []
    for s, i in zip(d_scores, d_idxs):
        if i == -1:
            continue
        rec = dict(id_map[i]); rec["_row"] = i
        faiss_hits.append((float(s), rec))

    if not hybrid:
        out_dense = [(float(s), dict(id_map[r["_row"]])) for s, r in faiss_hits[:k]]
        # --- domain MUST filtering even in dense-only mode ---
        qfold = _fold(text)
        need_airport = any(t in qfold for t in ["airport", "brandenburg"])
        need_gentr   = ("gentrif" in qfold) or ("neukolln" in qfold) or ("neukölln" in text.lower())
        if not (need_airport or need_gentr):
            return out_dense

        airport_terms = ["airport", "ber", "brandenburg", "willy brandt", "flughafen", "bbi"]
        gentr_terms   = ["gentrif", "rent", "landlord", "eviction", "miet", "kaltmiete", "warmmiete", "umwandlung"]

        filtered = []
        for score, rec in out_dense:
            window = " ".join([
                rec.get("title") or "",
                rec.get("summary_text") or "",
                _chunk_head_local(rec.get("chunk_text") or "", 120),
            ])
            keep = True
            if need_airport:
                keep = _contains_any(window, airport_terms)
            if keep and need_gentr:
                keep = _contains_any(window, gentr_terms) or _contains_any(window, ["neukolln", "neukölln"])
            if keep:
                filtered.append((score, rec))
        return filtered if filtered else out_dense

    # ----- Keyword branch (BM25) with domain-aware expansion -----
    # Safe import in case bm25/expand weren’t added yet
    try:
        from .bm25 import BM25Index
        from .expand import expand_for_bm25
    except Exception:
        # If hybrid code isn’t available, gracefully fall back to dense-only
        return [(float(s), dict(id_map[r["_row"]])) for s, r in faiss_hits[:k]]

    # Richer text for BM25: title + summary + head of chunk
    bm25_corpus = []
    for r in id_map:
        title = r.get("title") or ""
        summ  = r.get("summary_text") or ""
        head  = _chunk_head_local(r.get("chunk_text") or "", 120)
        bm25_corpus.append(f"{title}\n{summ}\n{head}")

    bm = BM25Index(bm25_corpus)
    expanded_terms = expand_for_bm25(text)
    b_raw = bm.search(expanded_terms, k=pool)

    bm25_hits: List[Tuple[float, Dict[str, Any]]] = []
    for s, i in b_raw:
        rec = dict(id_map[i]); rec["_row"] = i
        bm25_hits.append((float(s), rec))

    # ----- Fuse with RRF -----
    fused = _rrf_merge(faiss_hits, bm25_hits, k=k, rrf_k=60)

    out: List[Tuple[float, Dict[str, Any]]] = []
    for score, row in fused:
        rec = dict(id_map[row])
        out.append((float(score), rec))

    # ----- Domain MUST filters (post-fusion) -----
    qfold = _fold(text)
    need_airport = any(t in qfold for t in ["airport", "brandenburg"])
    need_gentr   = ("gentrif" in qfold) or ("neukolln" in qfold) or ("neukölln" in text.lower())

    if not (need_airport or need_gentr):
        return out

    airport_terms = ["airport", "ber", "brandenburg", "willy brandt", "flughafen", "bbi"]
    gentr_terms   = ["gentrif", "rent", "landlord", "eviction", "miet", "kaltmiete", "warmmiete", "umwandlung"]

    filtered = []
    for score, rec in out:
        window = " ".join([
            rec.get("title") or "",
            rec.get("summary_text") or "",
            _chunk_head_local(rec.get("chunk_text") or "", 120),
        ])
        keep = True
        if need_airport:
            keep = _contains_any(window, airport_terms)
        if keep and need_gentr:
            keep = _contains_any(window, gentr_terms) or _contains_any(window, ["neukolln", "neukölln"])
        if keep:
            filtered.append((score, rec))

    # If everything got filtered (rare), fall back to unfitered fused list
    return filtered if filtered else out

