"""
Dense Retrieval (FAISS)

PURPOSE:
    Online FAISS index search wrapper. Loads pre-built FAISS index from disk
    and searches for top-K nearest neighbors using normalized inner product (cosine similarity).

WHY:
    - In-memory index search takes < 1ms for 50k vectors.
    - Uses IndexFlatIP where higher score = greater similarity (cosine similarity).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    FAISS-based dense retriever.
    Loads a pre-built FAISS index from disk and executes vector search.
    """

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Run build_index first.")
        
        t0 = time.perf_counter()
        self.index = faiss.read_index(str(self.index_path))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Loaded FAISS index with %d vectors in %.2f ms", self.index.ntotal, elapsed)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search for top_k most similar chunks.
        
        Args:
            query_embedding: 2D numpy array of shape (1, dim), float32, L2-normalized.
            top_k: Number of candidates to retrieve.
            
        Returns:
            List of (chunk_index, score) tuples sorted descending by similarity.
        """
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # scores[0] is array of dot products, indices[0] is array of int indices
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
        return results
