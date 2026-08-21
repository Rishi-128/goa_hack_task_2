"""
Reciprocal Rank Fusion (RRF)

PURPOSE:
    Combine ranked lists from Dense (FAISS) and Sparse (BM25) retrieval into a unified ranking.

WHY THIS REPLACES SET UNION:
    Your original code did:
        hybrid_indices = list(set(list(indices) + list(bm_indices)))
    
    Problems with set union:
    1. It discards rank information completely. Document ranked #1 in both dense and sparse
       has the same priority as a document ranked #20 in just one.
    2. Python set ordering is non-deterministic, creating unstable context order.

    Why RRF is the industry standard for hybrid retrieval:
    - Score: RRF_score(d) = sum(1 / (k + rank_i(d)))
    - Standard k = 60 prevents top-ranked items from overly dominating.
    - Documents found by BOTH dense and sparse get a significant boost.
    - Scale-invariant: No need to normalize BM25 scores vs Cosine similarities.
"""

from typing import Iterable


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """
    Perform Reciprocal Rank Fusion on multiple ranked candidate lists.

    Args:
        ranked_lists: List of ranked lists, where each list contains (chunk_index, score) tuples.
                      Items are assumed to be sorted in descending order of relevance.
        k: Smoothing constant. Default is 60 (standard in IR literature).
        top_n: Number of top fused candidates to return.

    Returns:
        List of (chunk_index, rrf_score) tuples sorted descending by RRF score.
    """
    rrf_scores: dict[int, float] = {}

    for ranking in ranked_lists:
        for rank, (doc_id, _raw_score) in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort documents by accumulated RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:top_n]
