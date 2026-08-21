"""
Recovery & Fallback Handlers

PURPOSE:
    Provides graceful degradation and controlled error responses.
    Ensures that an API timeout, malformed response, or missing index NEVER crashes the service.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_abstention_response(reason: str = "insufficient_information") -> dict:
    """Standardized abstention response when knowledge base lacks context."""
    return {
        "answer": "I don't have enough information in the available knowledge base to answer that.",
        "summary": "Information not available in knowledge base.",
        "grounded": True,
        "sources": [],
    }


def create_safety_block_response(reason: str = "safety_filter") -> dict:
    """Standardized response when query violates safety rules."""
    return {
        "answer": "I cannot fulfill this request as it violates safety guidelines.",
        "summary": "Query blocked by safety policy.",
        "grounded": True,
        "sources": [],
    }


def create_error_fallback_response(error_message: str) -> dict:
    """Standardized fallback response on unexpected system or API errors."""
    logger.error("Pipeline fallback triggered: %s", error_message)
    return {
        "answer": "An error occurred while processing your request. Please try again.",
        "summary": "Service temporarily unavailable.",
        "grounded": False,
        "sources": [],
    }
