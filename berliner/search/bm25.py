import re
from __future__ import annotations
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import unicodedata

def _tokenize(s: str) -> List[str]:
    s = "".join(c for c in unicodedata.normalize("NFKD", (s or "").lower())
                if not unicodedata.combining(c))
    return [w for w in re.split(r"\W+", s) if w]

class BM25Index:
    def __init__(self, corpus_texts: List[str]):
        self.docs_tokenized = [_tokenize(t or "") for t in corpus_texts]
        self.bm25 = BM25Okapi(self.docs_tokenized)

    def search(self, query_terms: List[str], k: int = 50) -> List[Tuple[float, int]]:
        # Join expansion tokens into a single pseudo-query doc
        q_tokens = _tokenize(" ".join(query_terms))
        scores = self.bm25.get_scores(q_tokens)
        # top-k indices by score
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(float(scores[i]), i) for i in idxs]
