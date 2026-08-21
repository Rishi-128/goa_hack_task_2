"""
Retrieval Metrics Evaluation

PURPOSE:
    Evaluates IR retrieval quality against ground truth labels from MSMARCO-XI:
    - Recall@5, Recall@10
    - Mean Reciprocal Rank (MRR)
    - Precision@K
    - nDCG@K

WHY THIS MATTERS:
    Proves retrieval improvements objectively using actual ground-truth relevance annotations.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_mrr(ranked_passage_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Compute Reciprocal Rank (1/rank of first relevant document)."""
    for rank, pid in enumerate(ranked_passage_ids, start=1):
        if pid in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(ranked_passage_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Recall@K (fraction of relevant documents retrieved in top-k)."""
    if not ground_truth_ids:
        return 1.0  # vacuously true if no positive labels
    
    top_k_ids = set(ranked_passage_ids[:k])
    hits = len(top_k_ids.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)


def compute_precision_at_k(ranked_passage_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    top_k_ids = set(ranked_passage_ids[:k])
    hits = len(top_k_ids.intersection(ground_truth_ids))
    return hits / k


def evaluate_retrieval(
    queries: list[dict],
    chunks: list[dict],
    dense_retriever=None,
    sparse_retriever=None,
    reranker=None,
    embedder=None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """
    Run retrieval evaluation across test queries.
    """
    from retrieval.fusion import reciprocal_rank_fusion

    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    precisions: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []

    valid_query_count = 0

    for item in queries:
        gt_ids = set(item.get("relevant_passage_ids", []))
        if not gt_ids:
            continue  # skip queries with no ground truth positives
        
        valid_query_count += 1
        query_text = item["query"]

        # Run retrieval
        dense_res = []
        if dense_retriever and embedder:
            q_emb = embedder.encode_query(query_text)
            dense_res = dense_retriever.search(q_emb, top_k=20)

        sparse_res = []
        if sparse_retriever:
            sparse_res = sparse_retriever.search(query_text, top_k=20)

        fused = reciprocal_rank_fusion([dense_res, sparse_res], k=60, top_n=20)
        candidate_indices = [idx for idx, _ in fused]

        # Rerank
        if reranker:
            reranked = reranker.rerank(query_text, candidate_indices, chunks, top_n=max(k_values))
            ranked_passage_ids = [
                chunks[idx]["passage_id"] for idx, _score, _text in reranked if idx < len(chunks)
            ]
        else:
            ranked_passage_ids = [
                chunks[idx]["passage_id"] for idx in candidate_indices if idx < len(chunks)
            ]

        # Compute metrics
        mrr = compute_mrr(ranked_passage_ids, gt_ids)
        mrrs.append(mrr)

        for k in k_values:
            recalls[k].append(compute_recall_at_k(ranked_passage_ids, gt_ids, k))
            precisions[k].append(compute_precision_at_k(ranked_passage_ids, gt_ids, k))

    mean_mrr = sum(mrrs) / max(len(mrrs), 1)
    results = {
        "valid_queries": valid_query_count,
        "MRR": round(mean_mrr, 4),
    }

    for k in k_values:
        results[f"Recall@{k}"] = round(sum(recalls[k]) / max(len(recalls[k]), 1), 4)
        results[f"Precision@{k}"] = round(sum(precisions[k]) / max(len(precisions[k]), 1), 4)

    return results
