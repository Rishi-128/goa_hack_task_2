"""
Pipeline Guardrails

PURPOSE:
    Low-latency safety, domain relevance, retrieval confidence, and grounding validation.

WHY LOW-LATENCY MATTERS:
    Guardrails must NOT add another slow LLM call before or after generation.
    We implement deterministic, sub-5ms checks:
    1. Input Safety: Prompt injection / jailbreak / malicious input pattern filter.
    2. Domain Relevance / Off-topic: Max similarity threshold check against indexed embeddings.
    3. Retrieval Confidence: Reranker score gating to trigger controlled abstention.
    4. Grounding Validator: Post-generation lexical / entity grounding check against context.
"""

import logging
import re
import string
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Prompt injection & jailbreak patterns
UNSAFE_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"(system\s+prompt|reveal\s+instructions?|developer\s+mode)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?DAN|jailbreak", re.IGNORECASE),
    re.compile(r"format\s+c:|rm\s+-rf|<script.*?>", re.IGNORECASE),
]

# Stopwords for lightweight lexical overlap checking
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def check_input_safety(query: str) -> tuple[bool, Optional[str]]:
    """
    Check if the user input contains adversarial / unsafe patterns.

    Returns:
        (is_safe: bool, reason: Optional[str])
    """
    if not query or len(query.strip()) == 0:
        return False, "Empty query"

    if len(query) > 2000:
        return False, "Query exceeds maximum permitted length"

    for pattern in UNSAFE_PATTERNS:
        if pattern.search(query):
            logger.warning("Unsafe input pattern detected: %s", pattern.pattern)
            return False, "Input flagged by safety filter"

    return True, None


def check_domain_relevance(top_dense_score: float, threshold: Optional[float] = None) -> bool:
    """
    Check if query is relevant to the indexed knowledge base.
    Uses the top dense cosine similarity score.
    """
    th = threshold if threshold is not None else settings.offtopic_threshold
    return top_dense_score >= th


def check_retrieval_confidence(
    top_reranker_score: float,
    threshold: Optional[float] = None,
) -> bool:
    """
    Check if the reranker score indicates sufficient confidence to generate an answer.
    
    If False, the system should abstain rather than hallucinate.
    """
    th = threshold if threshold is not None else settings.confidence_threshold
    return top_reranker_score >= th


def check_grounding(
    answer: str,
    contexts: list[str],
    threshold: Optional[float] = None,
) -> tuple[bool, float]:
    """
    Post-generation validation: verifies if the generated answer is grounded in retrieved context.
    
    Computes content word overlap between answer tokens and context tokens.
    
    Returns:
        (is_grounded: bool, grounding_score: float)
    """
    th = threshold if threshold is not None else settings.grounding_threshold

    if not answer or not contexts:
        return False, 0.0

    # If the model explicitly stated it cannot find the content, consider it handled
    if "content not found" in answer.lower() or "not enough information" in answer.lower():
        return True, 1.0

    # Extract non-stopword tokens from answer
    answer_clean = re.sub(r"[^\w\s]", " ", answer.lower())
    answer_tokens = set(answer_clean.split()) - STOPWORDS
    
    if not answer_tokens:
        return True, 1.0

    # Extract all tokens from retrieved contexts
    context_text = " ".join(contexts).lower()
    context_clean = re.sub(r"[^\w\s]", " ", context_text)
    context_tokens = set(context_clean.split())

    # Calculate token recall from context
    grounded_tokens = answer_tokens.intersection(context_tokens)
    overlap_ratio = len(grounded_tokens) / len(answer_tokens)

    is_grounded = overlap_ratio >= th
    logger.debug("Grounding validation: overlap=%.2f (threshold=%.2f, pass=%s)", overlap_ratio, th, is_grounded)
    return is_grounded, overlap_ratio


def create_safety_block_response(reason: str = "safety_filter") -> dict:
    """Controlled response for blocked inputs."""
    if reason == "Empty query":
        return {
            "answer": "I didn't quite catch that. Could you please try again?",
            "summary": "Audio was not recognized.",
            "grounded": True,
        }
    return {
        "answer": "I cannot answer this question as it triggered our safety guidelines.",
        "summary": "Query blocked by safety filter.",
        "grounded": True,
    }


def create_abstention_response(reason: str = "low_retrieval_confidence") -> dict:
    """Controlled abstention response when retrieval confidence is too low."""
    return {
        "answer": "Content not found in the indexed knowledge base.",
        "summary": "No relevant documents found.",
        "grounded": True,
    }
