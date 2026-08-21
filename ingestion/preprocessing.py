"""
Text Preprocessing

PURPOSE:
    Normalize text before embedding and BM25 tokenization.
    
WHY:
    The original code did `text.split()` for BM25 tokenization.
    That loses information (no lowercasing, no punctuation handling).
    
    This module provides:
    1. General text normalization (whitespace, encoding)
    2. BM25-specific tokenization (lowercase, strip punctuation, etc.)
    
    Kept simple — don't over-engineer tokenization without measuring impact.
"""

import re
import string
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Basic text normalization applied to all passages before embedding.
    
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace into single space
    - Remove null bytes and control characters
    """
    if not text:
        return ""
    
    # Remove null bytes and control chars (except newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    
    Improvements over the original `text.split()`:
    1. Lowercase for case-insensitive matching
    2. Strip punctuation (but preserve important tokens like numbers)
    3. Normalize whitespace
    4. Filter empty tokens
    
    WHY NOT a full NLP tokenizer?
    - BM25 works well with simple whitespace tokenization
    - Adding NLTK/spaCy would increase latency and dependencies
    - We measured: this simple approach + BM25 is already effective
    - The reranker handles semantic nuance anyway
    """
    if not text:
        return []
    
    # Lowercase
    text = text.lower()
    
    # Replace punctuation with spaces (preserves word boundaries)
    # Keep hyphens inside words (e.g., "well-known"), apostrophes (e.g., "don't")
    text = re.sub(r"[^\w\s'-]", ' ', text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Filter: remove pure punctuation tokens and very short tokens
    tokens = [
        t.strip(string.punctuation)
        for t in tokens
        if len(t.strip(string.punctuation)) > 0
    ]
    
    return tokens
