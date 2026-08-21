"""
Pipeline State

PURPOSE:
    Defines the explicit state passed through the LangGraph RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
import numpy as np


class RAGStateDict(TypedDict, total=False):
    """TypedDict representation of pipeline state for LangGraph integration."""
    request_id: str
    original_query: str
    processed_query: str
    query_mode: str
    conversation_history: list[dict]
    
    # Validation & Guardrails
    is_safe: bool
    safety_error: Optional[str]
    is_domain_relevant: bool
    is_confident: bool
    is_grounded: bool
    should_abstain: bool
    abstain_reason: Optional[str]
    
    # Retrieval Data
    query_embedding: Optional[Any]
    dense_candidates: list[tuple[int, float]]
    sparse_candidates: list[tuple[int, float]]
    fused_candidates: list[tuple[int, float]]
    reranked_chunks: list[tuple[int, float, str]]
    top_reranker_score: float
    context_chunks: list[str]
    sources: list[dict]
    
    # Generation Data
    answer: str
    summary: str
    grounding_score: float
    
    # Observability & Latency Breakdown (in milliseconds)
    latency_breakdown: dict[str, float]
    total_latency_ms: float
    errors: list[str]
