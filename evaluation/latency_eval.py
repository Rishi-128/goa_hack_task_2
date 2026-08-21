"""
Latency Evaluation & Benchmarking (P50 / P70 / P100)

PURPOSE:
    Profiles the complete RAG pipeline across a batch of test queries,
    measuring component-level and end-to-end latencies with high-precision timers (time.perf_counter).

WHY P50 / P70 / P100:
    A single query measurement is noisy and misleading due to caching or cold starts.
    We compute the 50th percentile (median), 70th percentile, and 100th percentile (worst case).
"""

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_percentiles(values: list[float]) -> tuple[float, float, float]:
    """Compute P50, P70, P100 for a list of latency measurements (in ms)."""
    if not values:
        return 0.0, 0.0, 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    p50 = float(np.percentile(sorted_vals, 50))
    p70 = float(np.percentile(sorted_vals, 70))
    p100 = float(max(sorted_vals))
    
    return p50, p70, p100


def run_latency_benchmark(
    pipeline,
    test_queries: list[str],
    warmup_queries: int = 3,
) -> dict:
    """
    Run latency profiling across test queries.

    Args:
        pipeline: RAGPipelineGraph instance.
        test_queries: List of query strings.
        warmup_queries: Number of warmup queries to discard before measuring.

    Returns:
        Summary dict containing percentiles for all components.
    """
    logger.info("Running warmup on %d queries...", warmup_queries)
    for q in test_queries[:warmup_queries]:
        try:
            pipeline.run(q)
        except Exception:
            pass

    queries_to_benchmark = test_queries[warmup_queries:]
    logger.info("Benchmarking latency on %d queries...", len(queries_to_benchmark))

    component_latencies: dict[str, list[float]] = {}
    total_latencies: list[float] = []

    for i, q in enumerate(queries_to_benchmark):
        t0 = time.perf_counter()
        result = pipeline.run(q)
        total_ms = (time.perf_counter() - t0) * 1000
        total_latencies.append(total_ms)

        for comp, comp_ms in result.get("latency_breakdown", {}).items():
            if comp not in component_latencies:
                component_latencies[comp] = []
            component_latencies[comp].append(comp_ms)

    # Compute percentiles
    p50_total, p70_total, p100_total = compute_percentiles(total_latencies)

    component_report = {}
    for comp, vals in component_latencies.items():
        p50, p70, p100 = compute_percentiles(vals)
        component_report[comp] = {
            "p50": round(p50, 2),
            "p70": round(p70, 2),
            "p100": round(p100, 2),
        }

    report = {
        "num_queries": len(queries_to_benchmark),
        "total_latency": {
            "p50": round(p50_total, 2),
            "p70": round(p70_total, 2),
            "p100": round(p100_total, 2),
        },
        "components": component_report,
    }

    # Print formatted summary table
    print_latency_report(report)
    return report


def print_latency_report(report: dict):
    """Print standard formatted latency report."""
    print("\n" + "=" * 60)
    print("           RAG LATENCY BENCHMARK REPORT")
    print("=" * 60)
    print(f"Queries Evaluated: {report['num_queries']}")
    print(f"Total Latency -> P50: {report['total_latency']['p50']:.2f} ms | P70: {report['total_latency']['p70']:.2f} ms | P100: {report['total_latency']['p100']:.2f} ms")
    print("-" * 60)
    print(f"{'Component':<25} {'P50 (ms)':<10} {'P70 (ms)':<10} {'P100 (ms)':<10}")
    print("-" * 60)
    for comp, metrics in report["components"].items():
        print(f"{comp:<25} {metrics['p50']:<10.2f} {metrics['p70']:<10.2f} {metrics['p100']:<10.2f}")
    print("=" * 60 + "\n")
