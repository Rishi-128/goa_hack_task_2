"""
CrossEncoder Reranker

PURPOSE:
    Reranks top candidate chunks from hybrid retrieval using a Cross-Encoder model.

WHY:
    - Bi-encoders (SentenceTransformers) encode query and passages separately (fast retrieval).
    - Cross-encoders encode (query, passage) jointly with full cross-attention (much higher accuracy).
    - Reranking the top 10-20 hybrid candidates yields substantial MRR and Precision gains.

IMPROVEMENT OVER ORIGINAL:
    - Original code used `batch_size=1` which ran single-pair inferences sequentially.
    - We use batched prediction (`batch_size=16/32`) and support preloaded models.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder based reranker.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info("Loading CrossEncoder reranker: %s", model_name)
        t0 = time.perf_counter()
        self.model = CrossEncoder(model_name, max_length=max_length)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("CrossEncoder loaded in %.2f ms", elapsed)

    def rerank(
        self,
        query: str,
        candidate_indices: list[int],
        chunks: list[dict],
        top_n: int = 3,
    ) -> list[tuple[int, float, str]]:
        """
        Rerank candidates based on full cross-attention score.

        Args:
            query: Query string.
            candidate_indices: List of candidate chunk indices (from RRF).
            chunks: Full chunks list (with 'text' field).
            top_n: Number of top reranked chunks to return.

        Returns:
            List of (chunk_index, reranker_score, chunk_text) sorted descending by score.
        """
        if not candidate_indices:
            return []

        pairs = [
            [query, chunks[idx]["text"]]
            for idx in candidate_indices
            if idx < len(chunks)
        ]

        if not pairs:
            return []

        t0 = time.perf_counter()
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Reranked %d pairs in %.2f ms", len(pairs), elapsed)

        # Pair scores with candidate indices and chunk texts
        scored_candidates = [
            (candidate_indices[i], float(scores[i]), chunks[candidate_indices[i]]["text"])
            for i in range(len(pairs))
        ]

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_n]
