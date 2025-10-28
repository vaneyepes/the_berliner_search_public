# The berliner search indexer
# ======================================================
# FAISS index per embedding model
# ======================================================
# Purpose:
#   • Read embeddings from data/embeddings/<model-slug>/all_embeddings.npy
#   • Build a FAISS index (FlatIP for cosine search)
#   • Save results under data/index/<model-slug>/
# ======================================================

from __future__ import annotations
from pathlib import Path
import json
import time
from typing import List, Dict
import numpy as np
import faiss

#  reuse the Embedder class for encoding queries if needed later
from .embedder import Embedder

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def _slug(model_name: str) -> str:
    """Turn model name into a filesystem-safe folder name."""
    return model_name.replace("/", "__")


def build_index(
    chunks: List[Dict],              # [{chunk_id, text}, ...]
    model_name: str,
    embeddings_root: Path,
    index_root: Path,
    batch_size: int = 64,
) -> Dict[str, str]:
    """
    Build and save a FAISS index for the specified model.

    Steps:
      1. Encode each chunk's text using the given model.
      2. Save embeddings to data/embeddings/<model-slug>/all_embeddings.npy
      3. Build a FAISS IndexFlatIP (inner product for cosine).
      4. Save the index and ids under data/index/<model-slug>/.

    Returns a dict with useful paths for logging.
    """
    model_slug = _slug(model_name)
    emb_dir = Path(embeddings_root) / model_slug
    idx_dir = Path(index_root) / model_slug
    emb_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    print(f"[indexer] Encoding {len(texts)} chunks with {model_name} ...")
    embedder = Embedder(model_name, batch_size=batch_size)
    t0 = time.time()
    X = embedder.encode(texts)  # normalized embeddings
    elapsed = round(time.time() - t0, 2)
    print(f"[indexer] Done encoding in {elapsed}s")

    # Save embeddings (for reuse)
    np.save(emb_dir / "all_embeddings.npy", X)

    # Build FAISS index
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, str(idx_dir / "faiss.index"))

    # Save ids.jsonl (one per vector)
    with (idx_dir / "ids.jsonl").open("w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")

    # Save simple stats.json
    stats = {
        "model_name": model_name,
        "dim": d,
        "n_vectors": len(ids),
        "elapsed_sec": elapsed,
    }
    with (idx_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[indexer] Index written to {idx_dir/'faiss.index'}")
    print(f"[indexer] Vectors: {len(ids)} | Dim: {d}")

    return {
        "embeddings_path": str(emb_dir / "all_embeddings.npy"),
        "index_path": str(idx_dir / "faiss.index"),
        "ids_path": str(idx_dir / "ids.jsonl"),
        "stats_path": str(idx_dir / "stats.json"),
    }
