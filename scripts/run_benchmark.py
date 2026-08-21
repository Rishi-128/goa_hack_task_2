"""
Benchmark CLI

Runs benchmarks across the pipeline:
1. Retrieval IR Metrics (Recall@K, Precision@K, MRR)
2. Latency Benchmarks (P50, P70, P100 component & total)
3. Chunking Strategy Comparison

USAGE:
    python -m scripts.run_benchmark --type retrieval
    python -m scripts.run_benchmark --type latency --num-queries 50
    python -m scripts.run_benchmark --type all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from evaluation.latency_eval import run_latency_benchmark
from evaluation.retrieval_eval import evaluate_retrieval
from pipeline.graph import RAGPipelineGraph

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run RAG Benchmarks")
    parser.add_argument("--type", default="all", choices=["retrieval", "latency", "all"],
                        help="Benchmark type to execute")
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Number of queries to benchmark")
    parser.add_argument("--indexes-dir", default=None,
                        help="Path to indexes directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    indexes_dir = Path(args.indexes_dir or settings.indexes_dir)
    queries_path = indexes_dir / "queries.json"
    chunks_path = indexes_dir / "chunks.json"

    if not queries_path.exists() or not chunks_path.exists():
        print(f"\nERROR: Required index files not found in {indexes_dir}. Run `python -m scripts.build_index` first.\n")
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    test_queries = queries[:args.num_queries]
    query_texts = [q["query"] for q in test_queries if q.get("query")]

    print("\n" + "=" * 60)
    print(f"  STARTING BENCHMARK (Sample: {len(test_queries)} queries)")
    print("=" * 60)

    # 1. Retrieval Benchmark
    if args.type in ("retrieval", "all"):
        print("\n--- Running Retrieval Quality Evaluation (Recall@K, MRR) ---")
        pipeline = RAGPipelineGraph(indexes_dir=indexes_dir)
        retrieval_results = evaluate_retrieval(
            queries=test_queries,
            chunks=chunks,
            dense_retriever=pipeline.dense_retriever,
            sparse_retriever=pipeline.sparse_retriever,
            reranker=pipeline.reranker,
            embedder=pipeline.query_processor.embedder,
        )
        print("\n" + "=" * 50)
        print("       RETRIEVAL QUALITY RESULTS")
        print("=" * 50)
        for k, v in retrieval_results.items():
            print(f"  {k:<20}: {v}")
        print("=" * 50)

    # 2. Latency Benchmark
    if args.type in ("latency", "all"):
        print("\n--- Running Latency Benchmark (P50 / P70 / P100) ---")
        pipeline = RAGPipelineGraph(indexes_dir=indexes_dir)
        run_latency_benchmark(pipeline, query_texts, warmup_queries=2)


if __name__ == "__main__":
    main()
