"""
Chunking Strategy Evaluation

PURPOSE:
    Empirically benchmarks different chunking strategies against retrieval quality metrics:
    - Passthrough (baseline for MSMARCO)
    - Fixed Token Chunking (256, 512)
    - Recursive Chunking
    - Sentence Chunking
    - Semantic Chunking

OUTPUT:
    Generates a Markdown comparison table:
    Strategy              Recall@5   Recall@10   MRR      Latency
    -------------------------------------------------------------
"""

import json
import logging
import time
from pathlib import Path

from config.settings import settings
from evaluation.retrieval_eval import evaluate_retrieval
from ingestion.chunking.base import ChunkingStrategy
from ingestion.chunking.fixed import FixedChunkingStrategy
from ingestion.chunking.passthrough import PassthroughStrategy if "PassthroughStrategy" in locals() else None
from ingestion.chunking.recursive import RecursiveChunkingStrategy
from ingestion.chunking.semantic import SemanticChunkingStrategy
from ingestion.chunking.sentence import SentenceChunkingStrategy
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


def run_chunking_comparison(
    passages: list[dict],
    queries: list[dict],
    strategies: list[ChunkingStrategy],
    embedder: EmbeddingGenerator,
) -> list[dict]:
    """
    Evaluates multiple chunking strategies on the same dataset.
    """
    import faiss
    from rank_bm25 import BM25Okapi
    from ingestion.preprocessing import tokenize_for_bm25
    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever

    results = []

    for strategy in strategies:
        logger.info("Evaluating chunking strategy: %s", strategy.name)
        t0 = time.perf_counter()

        # 1. Chunk passages
        all_chunks = []
        for p in passages:
            meta = {
                "passage_id": p["passage_id"],
                "query_id": p["query_id"],
                "language": p.get("language", "eng"),
                "is_selected": p.get("is_selected", 0),
            }
            chunks = strategy.chunk(p["text"], meta)
            all_chunks.extend(chunks)

        chunk_time = (time.perf_counter() - t0) * 1000

        # 2. Embed chunks
        t_emb = time.perf_counter()
        chunk_texts = [c.text for c in all_chunks]
        embeddings = embedder.encode(chunk_texts, batch_size=256, show_progress=False)
        emb_time = (time.perf_counter() - t_emb) * 1000

        # 3. Build in-memory FAISS & BM25
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        tokenized = [tokenize_for_bm25(t) for t in chunk_texts]
        bm25 = BM25Okapi(tokenized)

        # 4. Wrap in temp retrievers
        class TempDense:
            def __init__(self, idx): self.index = idx
            def search(self, q_emb, top_k=20):
                scores, indices = self.index.search(q_emb.astype("float32"), top_k)
                return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]

        class TempSparse:
            def __init__(self, model): self.bm25 = model
            def search(self, query, top_k=20):
                tokens = tokenize_for_bm25(query)
                if not tokens: return []
                scores = self.bm25.get_scores(tokens)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                return [(i, float(scores[i])) for i in top_idx if scores[i] > 0]

        temp_dense = TempDense(index)
        temp_sparse = TempSparse(bm25)

        chunks_dict = [
            {"text": c.text, "passage_id": c.passage_id, "chunk_id": c.chunk_id}
            for c in all_chunks
        ]

        # 5. Evaluate retrieval
        eval_res = evaluate_retrieval(
            queries=queries,
            chunks=chunks_dict,
            dense_retriever=temp_dense,
            sparse_retriever=temp_sparse,
            embedder=embedder,
        )

        results.append({
            "strategy": strategy.name,
            "num_chunks": len(all_chunks),
            "Recall@5": eval_res.get("Recall@5", 0.0),
            "Recall@10": eval_res.get("Recall@10", 0.0),
            "MRR": eval_res.get("MRR", 0.0),
            "Precision@5": eval_res.get("Precision@5", 0.0),
            "chunk_latency_ms": round(chunk_time, 2),
        })

    return results


def print_chunking_report(results: list[dict]):
    """Print formatted Markdown table comparison of chunking strategies."""
    print("\n" + "=" * 70)
    print(f"{'Strategy':<20} {'Chunks':<10} {'Recall@5':<12} {'Recall@10':<12} {'MRR':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<20} {r['num_chunks']:<10} {r['Recall@5']:<12.4f} {r['Recall@10']:<12.4f} {r['MRR']:<10.4f}")
    print("=" * 70 + "\n")
