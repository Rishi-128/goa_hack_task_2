"""
Semantic Chunking

Groups consecutive sentences by embedding similarity.
When similarity drops below a threshold, a new chunk starts.

WHY:
    Fixed and sentence-based chunking ignore content boundaries.
    Semantic chunking detects topic shifts and splits there.

COST:
    Requires embedding every sentence — expensive but this is OFFLINE ONLY.
    We pre-compute once and save to disk, so latency doesn't matter here.
"""

import re
import logging

import numpy as np

from .base import Chunk, ChunkingStrategy

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Group consecutive sentences by embedding similarity.
    
    Algorithm:
        1. Split text into sentences
        2. Embed each sentence
        3. Compare consecutive sentence embeddings (cosine similarity)
        4. When similarity drops below threshold → start new chunk
        5. Merge small chunks with neighbors
    
    Args:
        model_name: SentenceTransformer model for sentence embeddings
        threshold: Similarity threshold for splitting (0.0 - 1.0)
        min_sentences: Minimum sentences per chunk
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_sentences: int = 2,
    ):
        self.threshold = threshold
        self.min_sentences = min_sentences
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        """Lazy-load the model (only needed during offline indexing)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"semantic_{self.threshold}"

    def _split_sentences(self, text: str) -> list[str]:
        sentences = _SENTENCE_SPLIT.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        sentences = self._split_sentences(text)

        # If too few sentences, return as-is
        if len(sentences) <= self.min_sentences:
            return [
                Chunk(
                    text=text,
                    chunk_id=f"{metadata['passage_id']}_chunk_0",
                    passage_id=metadata["passage_id"],
                    query_id=metadata["query_id"],
                    chunk_index=0,
                    chunk_strategy=self.name,
                    language=metadata.get("language", ""),
                    is_selected=metadata.get("is_selected", 0),
                )
            ]

        # Embed all sentences
        model = self._get_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)

        # Find split points: where consecutive similarity drops below threshold
        split_points = []
        for i in range(len(embeddings) - 1):
            similarity = float(np.dot(embeddings[i], embeddings[i + 1]))
            if similarity < self.threshold:
                split_points.append(i + 1)

        # Group sentences into chunks
        chunk_texts = []
        prev = 0
        for sp in split_points:
            group = sentences[prev:sp]
            if len(group) >= 1:
                chunk_texts.append(" ".join(group))
            prev = sp
        # Last group
        remaining = sentences[prev:]
        if remaining:
            # Merge tiny trailing groups with previous chunk
            if chunk_texts and len(remaining) < self.min_sentences:
                chunk_texts[-1] += " " + " ".join(remaining)
            else:
                chunk_texts.append(" ".join(remaining))

        if not chunk_texts:
            chunk_texts = [text]

        return [
            Chunk(
                text=t,
                chunk_id=f"{metadata['passage_id']}_chunk_{i}",
                passage_id=metadata["passage_id"],
                query_id=metadata["query_id"],
                chunk_index=i,
                chunk_strategy=self.name,
                language=metadata.get("language", ""),
                is_selected=metadata.get("is_selected", 0),
            )
            for i, t in enumerate(chunk_texts)
        ]
