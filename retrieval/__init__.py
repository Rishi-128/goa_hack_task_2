"""
Retrieval package.

Modules:
    - DenseRetriever: In-memory FAISS IndexFlatIP cosine search
    - SparseRetriever: BM25Okapi keyword search with argpartition top-k
    - reciprocal_rank_fusion: Rank-based score fusion (RRF)
    - CrossEncoderReranker: Batched Cross-Encoder reranker
"""

from .dense import DenseRetriever
from .sparse import SparseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker

__all__ = [
    "DenseRetriever",
    "SparseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
]
