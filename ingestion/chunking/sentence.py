"""
Sentence-based Chunking

Splits text on sentence boundaries and combines sentences
until a word limit is reached.

WHY:
    Fixed-size chunking can cut sentences mid-thought.
    Sentence-based chunking preserves semantic boundaries,
    which improves embedding quality.
"""

import re

from .base import Chunk, ChunkingStrategy


# Simple sentence splitting pattern
# Handles: period, question mark, exclamation mark followed by space or end
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


class SentenceChunkingStrategy(ChunkingStrategy):
    """
    Split on sentence boundaries, then combine sentences up to max_words.
    
    Args:
        max_words: Maximum words per chunk
        min_words: Minimum words per chunk (avoids tiny trailing chunks)
    """

    def __init__(self, max_words: int = 80, min_words: int = 20):
        self.max_words = max_words
        self.min_words = min_words

    @property
    def name(self) -> str:
        return f"sentence_{self.max_words}"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = _SENTENCE_SPLIT.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        sentences = self._split_sentences(text)

        if not sentences:
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
        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if current_word_count + word_count > self.max_words and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(chunk_text)
                current_sentences = [sentence]
                current_word_count = word_count
            else:
                current_sentences.append(sentence)
                current_word_count += word_count

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # If this trailing chunk is too small, merge with previous
            if chunks and current_word_count < self.min_words:
                chunks[-1] = chunks[-1] + " " + chunk_text
            else:
                chunks.append(chunk_text)

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
            for i, t in enumerate(chunks)
        ]
