"""
Unit Tests for Retrieval and Fusion Components
"""

import pytest
import numpy as np

from retrieval.fusion import reciprocal_rank_fusion
from ingestion.preprocessing import tokenize_for_bm25, normalize_text
from ingestion.chunking.fixed import FixedChunkingStrategy
from ingestion.chunking.sentence import SentenceChunkingStrategy


def test_tokenize_for_bm25():
    text = "What is a Corporation? It's an incorporated entity!"
    tokens = tokenize_for_bm25(text)
    assert "what" in tokens
    assert "corporation" in tokens
    assert "incorporated" in tokens
    assert "entity" in tokens
    # Verify punctuation is stripped
    assert "?" not in tokens
    assert "!" not in tokens


def test_reciprocal_rank_fusion():
    # Dense results: doc 0 (#1), doc 1 (#2), doc 2 (#3)
    dense = [(0, 0.95), (1, 0.85), (2, 0.75)]
    # Sparse results: doc 1 (#1), doc 0 (#2), doc 3 (#3)
    sparse = [(1, 12.5), (0, 10.0), (3, 8.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=4)
    
    # Doc 0 and Doc 1 appear in both lists, so they must be top 2
    top_docs = [doc_id for doc_id, score in fused[:2]]
    assert 0 in top_docs
    assert 1 in top_docs
    
    # Scores must be strictly descending
    scores = [score for doc_id, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_fixed_chunking():
    strategy = FixedChunkingStrategy(chunk_size=10, chunk_overlap=2)
    words = ["word" + str(i) for i in range(25)]
    text = " ".join(words)
    metadata = {"passage_id": "p1", "query_id": 100, "language": "eng", "is_selected": 1}

    chunks = strategy.chunk(text, metadata)
    assert len(chunks) > 1
    assert chunks[0].passage_id == "p1"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_sentence_chunking():
    strategy = SentenceChunkingStrategy(max_words=10, min_words=2)
    text = "This is the first sentence. Here is another sentence with more words. Finally a third sentence."
    metadata = {"passage_id": "p2", "query_id": 101, "language": "eng", "is_selected": 0}

    chunks = strategy.chunk(text, metadata)
    assert len(chunks) >= 1
    assert all(isinstance(c.text, str) and len(c.text) > 0 for c in chunks)
