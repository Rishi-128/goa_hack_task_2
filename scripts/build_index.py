"""
Build Index CLI

Runs the offline ingestion pipeline:
    Dataset → Chunk → Embed → Save FAISS + BM25 + metadata

USAGE:
    # Default: passthrough chunking, 10K sample, English
    python -m scripts.build_index

    # Custom configuration
    python -m scripts.build_index --sample 5000 --strategy recursive --chunk-size 512

    # Use settings from .env
    python -m scripts.build_index --use-config
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes")
    
    parser.add_argument("--output", default="indexes",
                        help="Output directory for indexes")
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XI",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", default="validation",
                        help="Dataset split")
    parser.add_argument("--languages", default=None,
                        help="Comma-separated target languages (e.g. 'hin_Deva,ben_Beng')")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Number of rows to sample (0 for all)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Embedding batch size")
    parser.add_argument("--strategy", default="passthrough",
                        choices=["passthrough", "fixed", "recursive", "sentence", "semantic"],
                        help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size (for fixed/recursive/sentence)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Chunk overlap (for fixed/recursive)")
    parser.add_argument("--semantic-threshold", type=float, default=0.5,
                        help="Similarity threshold for semantic chunking")
    parser.add_argument("--use-config", action="store_true",
                        help="Use values from config/settings.py instead of CLI args")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # If --use-config, override from settings
    if args.use_config:
        from config.settings import settings
        args.output = str(settings.indexes_dir)
        args.dataset = settings.dataset_name
        args.split = settings.dataset_split
        args.languages = settings.dataset_languages
        args.sample = settings.dataset_sample_size
        args.model = settings.embedding_model
        args.batch_size = settings.embedding_batch_size
        args.strategy = settings.chunking_strategy
        args.chunk_size = settings.chunk_size
        args.chunk_overlap = settings.chunk_overlap
        args.semantic_threshold = settings.semantic_threshold

    # Parse languages
    target_languages = None
    if args.languages:
        target_languages = [l.strip() for l in args.languages.split(",")]

    from ingestion.index_builder import build_index

    result = build_index(
        output_dir=args.output,
        dataset_name=args.dataset,
        split=args.split,
        target_languages=target_languages,
        sample_size=args.sample,
        embedding_model=args.model,
        embedding_batch_size=args.batch_size,
        normalize_embeddings=True,
        chunking_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        semantic_threshold=args.semantic_threshold,
    )

    print("\n" + "=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
