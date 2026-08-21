"""
Chunking sub-package.

Available strategies:
    - PassthroughStrategy: Use passages as-is (default for MSMARCO-XI)
    - FixedChunkingStrategy: Fixed word-count chunks with overlap
    - RecursiveChunkingStrategy: LangChain RecursiveCharacterTextSplitter
    - SentenceChunkingStrategy: Sentence-boundary-aware chunking
    - SemanticChunkingStrategy: Embedding-similarity-based chunking
"""

from .base import Chunk, ChunkingStrategy, PassthroughStrategy
from .fixed import FixedChunkingStrategy
from .recursive import RecursiveChunkingStrategy
from .sentence import SentenceChunkingStrategy
from .semantic import SemanticChunkingStrategy

__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "PassthroughStrategy",
    "FixedChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SentenceChunkingStrategy",
    "SemanticChunkingStrategy",
]
