"""
Sparse Retrieval (BM25)

PURPOSE:
    Online BM25 search wrapper. Loads pre-computed BM25 model from disk
    and searches for keyword matches using clean tokenization.

WHY:
    - Complements dense retrieval by catching exact keyword matches, acronyms, and specific numbers.
    - Improved tokenization (lowercased, punctuation handled) improves over naive string split.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.preprocessing import tokenize_for_bm25

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25-based sparse retriever.
    Loads pickled BM25Okapi model from disk and executes keyword search.
    """

    def __init__(self, bm25_path: str | Path):
        self.bm25_path = Path(bm25_path)
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.bm25_path}. Run build_index first.")
        
        t0 = time.perf_counter()
        with open(self.bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Loaded BM25 index with %d docs in %.2f ms", self.bm25.corpus_size, elapsed)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search for top_k most relevant chunks using BM25.
        
        Args:
            query: Raw query text.
            top_k: Number of candidates to retrieve.
            
        Returns:
            List of (chunk_index, bm25_score) tuples sorted descending by score.
        """
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # argpartition is faster than full sort for large corpora
            partition_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = partition_idx[np.argsort(scores[partition_idx])[::-1]]
        
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results
