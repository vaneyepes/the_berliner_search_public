# The berliner embedder module:
# goal: Load a sentence-transformers model by name (from config) and encode texts.


from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np

# We rely on sentence-transformers for high-quality retrieval encoders.
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Tiny wrapper around SentenceTransformer for consistency.
    - model_name: HF model id (e.g., 'sentence-transformers/multi-qa-mpnet-base-dot-v1')
    - batch_size: number of texts to encode per forward pass
    """
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        # SentenceTransformer handles device selection automatically (GPU if available).
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into a 2D float32 array [n_texts, dim].
        We use normalize_embeddings=True to help cosine similarities.
        """
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # good default for retrieval
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
