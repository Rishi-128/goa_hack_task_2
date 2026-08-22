"""
Query Processor

PURPOSE:
    Handles query preparation:
    1. Conditional query rewriting (skips LLM call when query is already self-contained)
    2. Optional HyDE (Hypothetical Document Embeddings) generation
    3. Query embedding generation
"""

import logging
import re
import time
from typing import Optional

import numpy as np

from config.settings import settings
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

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
        """
        if not conversation_history:
            return False

        if PRONOUN_PATTERN.search(query):
            return True

        if len(query.split()) <= 3:
            return True

        return False

    def rewrite_query(self, query: str, conversation_history: list) -> tuple[str, float]:
        """
        Rewrite query into a standalone question using conversation history.
        """
        t0 = time.perf_counter()
        prompt = f"""Conversation History:
{conversation_history}

Current User Question:
{query}

Rewrite the question into a standalone, fully self-contained query.
Do NOT answer the question. Output ONLY the rewritten question.
If no rewrite is needed, return the original query."""

        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
            )
            rewritten = response.choices[0].message.content.strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return (rewritten if rewritten else query), elapsed_ms
        except Exception as e:
            logger.warning("Query rewrite failed (%s), using original", e)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return query, elapsed_ms

    def generate_hyde_document(self, query: str) -> tuple[str, float]:
        """Generate hypothetical document for dense embedding search."""
        t0 = time.perf_counter()
        prompt = f"Please write a short passage that directly answers the question: {query}"
        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            hyde_doc = response.choices[0].message.content.strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return hyde_doc or query, elapsed_ms
        except Exception as e:
            logger.warning("HyDE generation failed (%s)", e)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return query, elapsed_ms

    def process(
        self,
        query: str,
        conversation_history: list,
        query_mode: str = "normal",
    ) -> dict:
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
