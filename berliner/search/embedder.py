"""
Embeddings generator for The Berliner Search (Phase 6).

- Loads records from data/enriched/metadata.jsonl
- Chooses text from 'summary_text' then fallback to 'chunk_text'
- Encodes with Sentence-Transformers (MiniLM L6 v2, 384-D)
- Optionally normalizes vectors for cosine similarity via inner product
- Saves:
    - data/index/embeddings.npy (float32)
    - data/index/ids.jsonl (one JSON per line, mapping vector row -> record ids)
    - data/index/stats.json (model, dim, counts, time, settings)
"""

from __future__ import annotations
from pathlib import Path
import json, time
from typing import Iterable, List, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]  # THE_BERLINER_SEARCH/
DATA = ROOT / "data"
ENRICHED = DATA / "enriched"
INDEX = DATA / "index"

METADATA_PATH = ENRICHED / "metadata.jsonl"
EMBEDDINGS_PATH = INDEX / "embeddings.npy"
IDS_PATH = INDEX / "ids.jsonl"
STATS_PATH = INDEX / "stats.json"

# Defaults (can be overridden by config.yaml if desired)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FIELD_PRIORITY = ["summary_text", "chunk_text"]
NORMALIZE = True
BATCH_SIZE = 64
MAX_LENGTH = 512  # tokens handled internally by the model with truncation

def ensure_dirs() -> None:
    INDEX.mkdir(parents=True, exist_ok=True)

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def choose_text(rec):
    # Prefer summary, but enrich with a short slice of the chunk (boosts recall on specific entities like "BER")
    summary = (rec.get("summary_text") or "").strip()
    chunk   = (rec.get("chunk_text") or "").strip()
    chunk_head = " ".join(chunk.split()[:120]) if chunk else ""
    combo = (summary + ("\n\n" + chunk_head if chunk_head else "")).strip()
    return combo or None


def load_records(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Missing file: {path}"
    return [r for r in iter_jsonl(path)]

def to_batches(items: List[str], bs: int) -> Iterable[List[str]]:
    for i in range(0, len(items), bs):
        yield items[i : i + bs]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    # Avoid div-by-zero by adding small epsilon
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norms).astype("float32")

def encode_texts(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """
    Encodes texts into embeddings (float32), on CPU or GPU if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    all_vecs: List[np.ndarray] = []

    # encode already does batching internally, but we keep explicit control for clarity
    for batch in tqdm(to_batches(texts, batch_size), total=(len(texts)+batch_size-1)//batch_size, desc="Encoding"):
        # convert_to_numpy=True returns float32
        vecs = model.encode(
            batch,
            batch_size=len(batch),  # the library handles internal micro-batching
            convert_to_numpy=True,
            normalize_embeddings=False,  # we handle normalization ourselves if needed
            show_progress_bar=False,
        )
        all_vecs.append(vecs)

    return np.vstack(all_vecs).astype("float32")

def run(model_name: str = MODEL_NAME,
        normalize: bool = NORMALIZE,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH) -> Dict[str, Any]:
    t0 = time.time()
    ensure_dirs()

    records = load_records(METADATA_PATH)

       # Build text list + id map
    texts: List[str] = []
    id_map: List[Dict[str, Any]] = []
    for rec in records:
        txt = choose_text(rec)
        if not txt:
            continue

        # Be robust to schema variations
        pages = rec.get("page_span") or rec.get("pages") or rec.get("page_range")
        title = rec.get("title") or rec.get("headline") or rec.get("section_title")

        texts.append(txt)
        id_map.append({
            "issue_id": rec.get("issue_id"),
            "chunk_id": rec.get("chunk_id"),
            "page_span": pages,      # keep the same printed key used by CLI
            "title": title,
            "has_summary": bool(rec.get("summary_text")),
        })

    if not texts:
        raise RuntimeError("No valid texts found to embed. Check metadata.jsonl fields.")

    # Encode
    embeddings = encode_texts(texts, model_name, batch_size)

    # Normalize for cosine (so IP == cosine)
    if normalize:
        embeddings = l2_normalize(embeddings)

    # Save artifacts
    np.save(EMBEDDINGS_PATH, embeddings)
    with IDS_PATH.open("w", encoding="utf-8") as f:
        for row in id_map:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = dict(
        model_name=model_name,
        dim=int(embeddings.shape[1]),
        n_vectors=int(embeddings.shape[0]),
        normalize=normalize,
        batch_size=batch_size,
        elapsed_sec=round(time.time() - t0, 2),
    )
    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats

if __name__ == "__main__":
    s = run()
    print("[embedder] done:", s)
