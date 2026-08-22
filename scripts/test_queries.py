"""
Multi-Query Quality Test across 23M+ Passages
"""

import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.graph import RAGPipelineGraph


def run_tests():
    graph = RAGPipelineGraph()

    test_queries = [
        "What is a corporation?",
        "what is the barter system and its main problem?",
        "symptoms of borderline personality disorder",
        "how many chromosomes do humans have?",
        "what are gamma rays and radiation?",
        "who wrote the obligation to endure?",
        "home remedies for pimples",
        "distance from scottsdale to grand canyon",
    ]

    print("=" * 70)
    print("🧪 MULTI-QUERY RAG VERIFICATION REPORT")
    print("=" * 70)

    for q in test_queries:
        res = graph.run(q)
        ans = res.get("answer", "")
        src = res.get("sources", [])
        lat = res.get("total_latency_ms", 0.0)
        src_id = src[0]["passage_id"] if src else "None"

        print(f"\n▶ Query: \"{q}\"")
        print(f"  ⚡ Latency: {lat:.1f} ms")
        print(f"  📄 Top Source: {src_id}")
        print(f"  💬 Answer: {ans}")
        print("-" * 70)


if __name__ == "__main__":
    run_tests()
