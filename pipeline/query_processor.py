"""
Query Processor

PURPOSE:
    Handles query preparation:
    1. Conditional query rewriting (skips LLM call when query is already self-contained)
    2. Optional HyDE (Hypothetical Document Embeddings) generation
    3. Query embedding generation

WHY CONDITIONAL REWRITING IS ESSENTIAL:
    - Calling an LLM to rewrite EVERY query adds 300-600 ms of pure overhead.
    - For standalone questions ("what is a corporation?"), rewrite is completely redundant.
    - We check: Is conversation history non-empty? Are there unresolved pronouns ("it", "they", "its", etc.)?
    - If NO -> bypass LLM rewrite (saves ~400ms, essential for < 200ms target).
    - If YES -> run LLM rewrite (preserving the prompt from rag_core.py).
"""

import logging
import re
import time
from typing import Optional

import numpy as np

from config.settings import settings
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Patterns indicating unresolved conversational references / pronouns
PRONOUN_PATTERN = re.compile(
    r"\b(it|its|they|them|their|theirs|he|him|his|she|her|hers|this|that|these|those|the former|the latter)\b",
    re.IGNORECASE,
)


class QueryProcessor:
    """
    Handles query analysis, conditional rewriting, HyDE generation, and embedding.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingGenerator] = None,
        llm_client=None,
    ):
        self.embedder = embedder or EmbeddingGenerator(
            model_name=settings.embedding_model,
            normalize=settings.normalize_embeddings,
        )
        self.llm_client = llm_client

    def _get_llm_client(self):
        if self.llm_client is None:
            from groq import Groq
            self.llm_client = Groq(api_key=settings.groq_api_key)
        return self.llm_client

    def needs_rewrite(self, query: str, conversation_history: list) -> bool:
        """
        Heuristic check: does the query need context-dependent rewriting?
        
        Returns True if:
        1. There is active conversation history AND
        2. Query contains pronouns or references that indicate context dependency, OR is very short (< 4 words)
        """
        if not conversation_history:
            return False

        # If history exists and query has pronouns/anaphora
        if PRONOUN_PATTERN.search(query):
            return True

        # Short follow-ups like "why?", "how much?", "and then?"
        if len(query.split()) <= 3:
            return True

        return False

    def rewrite_query(self, query: str, conversation_history: list) -> tuple[str, float]:
        """
        Rewrite query into a standalone question using conversation history.
        Uses exact prompt logic from original rag_core.py.
        
        Returns:
            (rewritten_query, elapsed_ms)
        """
        t0 = time.perf_counter()
        prompt = f"""Conversation History:
{conversation_history}

Current User Question:
{query}

Rewrite the question into a standalone, fully self-contained query.
Do NOT answer the question.
If no rewrite is needed, return the original query."""

        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )

        except Exception as e:
            logger.warning("HyDE generation failed (%s), falling back to query", e)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return query, elapsed_ms

    def process(
        self,
        query: str,
        conversation_history: list,
        query_mode: str = "normal",
    ) -> dict:
        """
        Full query processing step:
        1. Condition-checked rewriting
        2. Optional HyDE
        3. Embedding generation
        
        Returns dict with:
            - processed_query: str
            - embedding_text: str (text used for embedding)
            - query_embedding: np.ndarray
            - rewrite_latency_ms: float
            - hyde_latency_ms: float
            - embedding_latency_ms: float
        """
        rewrite_latency_ms = 0.0
        hyde_latency_ms = 0.0
        processed_query = query

        # 1. Query Rewrite
        if query_mode == "query_rewrite" or (settings.auto_rewrite and self.needs_rewrite(query, conversation_history)):
            processed_query, rewrite_latency_ms = self.rewrite_query(query, conversation_history)

        # 2. HyDE
        embedding_text = processed_query
        if query_mode == "hyde":
            embedding_text, hyde_latency_ms = self.generate_hyde_document(processed_query)

        # 3. Query Embedding
        t0 = time.perf_counter()
        query_embedding = self.embedder.encode_query(embedding_text)
        embedding_latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "original_query": query,
            "processed_query": processed_query,
            "embedding_text": embedding_text,
            "query_embedding": query_embedding,
            "rewrite_latency_ms": rewrite_latency_ms,
            "hyde_latency_ms": hyde_latency_ms,
            "embedding_latency_ms": embedding_latency_ms,
        }
