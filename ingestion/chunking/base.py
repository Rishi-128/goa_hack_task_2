"""
Abstract Base for Chunking Strategies

Every chunking strategy takes a passage + metadata and produces
a list of chunks, each with its own metadata.

WHY ABSTRACT BASE:
    Makes strategies interchangeable for benchmarking.
    The chunking_eval.py can loop through all strategies uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    A single chunk produced by a chunking strategy.
    
    Every chunk carries metadata so we can trace it back to
    its source passage and evaluate retrieval quality.
    """
    text: str
    chunk_id: str           # Unique ID: f"{passage_id}_chunk_{index}"
    passage_id: str         # Source passage ID
    query_id: int | str     # Originating query ID
    chunk_index: int        # Position within the passage (0 if passthrough)
    chunk_strategy: str     # Which strategy produced this chunk
    language: str           # Language code
    is_selected: int        # Ground truth: 1 if source passage was relevant
    extra: dict = field(default_factory=dict)  # Any additional metadata


class ChunkingStrategy(ABC):
    """Abstract base class for all chunking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy."""
        ...

    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Split a passage into one or more chunks.
        
        Args:
            text: The passage text to chunk
            metadata: Dict with keys: passage_id, query_id, language, is_selected
            
        Returns:
            List of Chunk objects
        """
        ...


class PassthroughStrategy(ChunkingStrategy):
    """
    Use passages as-is without further splitting.
    
    WHY THIS IS THE DEFAULT:
        MSMARCO-XI passages are already 294 chars median (50-70 tokens).
        This is already in the sweet spot for embedding models.
        Further splitting would fragment meaning.
        Combining would bloat context.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
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
