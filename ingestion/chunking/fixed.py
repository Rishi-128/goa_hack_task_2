"""
Fixed-size Token Chunking

Splits text into chunks of a fixed token count with configurable overlap.
Tokens here = whitespace-split words (not subword tokens), which is
a reasonable approximation that avoids needing a tokenizer dependency.
"""

from .base import Chunk, ChunkingStrategy


class FixedChunkingStrategy(ChunkingStrategy):
    """
    Split text into fixed-size word chunks with overlap.
    
    Args:
        chunk_size: Number of words per chunk
        chunk_overlap: Number of overlapping words between chunks
    """

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        return f"fixed_{self.chunk_size}_{self.chunk_overlap}"

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        words = text.split()
        
        # If text is shorter than chunk_size, return as single chunk
        if len(words) <= self.chunk_size:
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

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        step = max(step, 1)  # Prevent infinite loop

        for i, start in enumerate(range(0, len(words), step)):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=f"{metadata['passage_id']}_chunk_{i}",
                    passage_id=metadata["passage_id"],
                    query_id=metadata["query_id"],
                    chunk_index=i,
                    chunk_strategy=self.name,
                    language=metadata.get("language", ""),
                    is_selected=metadata.get("is_selected", 0),
                )
            )

            if end >= len(words):
                break

        return chunks
