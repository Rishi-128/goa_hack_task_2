"""
Recursive Character Text Splitter

Wraps LangChain's RecursiveCharacterTextSplitter — this is the
BASELINE from your original code.

Preserved because:
1. It's what you already had (chunk_size=650, chunk_overlap=50)
2. It provides a known reference point for benchmarking
3. It's a well-tested splitting approach
"""

from .base import Chunk, ChunkingStrategy


class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    RecursiveCharacterTextSplitter wrapper.
    
    Splits on a hierarchy of separators: paragraphs → sentences → words.
    This is the exact approach from your original rag_core.py.
    """

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 50):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def name(self) -> str:
        return f"recursive_{self.chunk_size}_{self.chunk_overlap}"

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        # If text is shorter than chunk_size, return as-is
        if len(text) <= self.chunk_size:
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

        split_texts = self.splitter.split_text(text)

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
            for i, t in enumerate(split_texts)
        ]
