"""
Index Builder — Offline Pipeline

PURPOSE:
    Takes passages → chunks → embeddings → saves FAISS index + BM25 + chunk store.
    
THIS IS THE CRITICAL ARCHITECTURE CHANGE:
    Your original code did this EVERY time in loading():
        load files → chunk → embed → build index
    
    Now we do it ONCE and save to disk:
        load dataset → chunk → embed → save FAISS index
                                     → save BM25 corpus
                                     → save chunk metadata
    
    At serving time, we just load the pre-built indexes.
    This drops startup from minutes to seconds.

WHAT GETS SAVED:
    indexes/
    ├── faiss.index         # FAISS vector index
    ├── embeddings.npy      # Raw embeddings (for debugging/re-indexing)
    ├── bm25_corpus.json    # Tokenized texts for BM25
    ├── chunks.json         # Chunk texts + metadata
    └── queries.json        # Evaluation queries with ground truth
"""

import json
import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.chunking.base import Chunk, ChunkingStrategy, PassthroughStrategy
from ingestion.chunking.fixed import FixedChunkingStrategy
from ingestion.chunking.recursive import RecursiveChunkingStrategy
from ingestion.chunking.sentence import SentenceChunkingStrategy
from ingestion.chunking.semantic import SemanticChunkingStrategy
from ingestion.dataset import load_msmarco_passages
from ingestion.embeddings import EmbeddingGenerator
from ingestion.preprocessing import normalize_text, tokenize_for_bm25

logger = logging.getLogger(__name__)


STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "passthrough": PassthroughStrategy,
    "fixed": FixedChunkingStrategy,
    "recursive": RecursiveChunkingStrategy,
    "sentence": SentenceChunkingStrategy,
    "semantic": SemanticChunkingStrategy,
}


def get_strategy(name: str, **kwargs) -> ChunkingStrategy:
    """Get a chunking strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)


def build_index(
    output_dir: str | Path = "indexes",
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "validation",
    target_languages: list[str] | None = None,
    sample_size: int = 10000,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_batch_size: int = 256,
    normalize_embeddings: bool = True,
    chunking_strategy: str = "passthrough",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    semantic_threshold: float = 0.5,
):
    """
    Full offline indexing pipeline.
    
    1. Load dataset
    2. Chunk passages
    3. Generate embeddings
    4. Build FAISS index (IndexFlatIP with normalized embeddings)
    5. Build BM25 index
    6. Save everything to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_t0 = time.perf_counter()

    # ── Step 1: Load dataset ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading dataset")
    logger.info("=" * 60)

    passages, queries = load_msmarco_passages(
        dataset_name=dataset_name,
        split=split,
        target_languages=target_languages,
        sample_size=sample_size,
        use_english_passages=True,
    )

    # ── Step 2: Chunk passages ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking with strategy='%s'", chunking_strategy)
    logger.info("=" * 60)

    strategy_kwargs = {}
    if chunking_strategy == "fixed":
        strategy_kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    elif chunking_strategy == "recursive":
        strategy_kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    elif chunking_strategy == "sentence":
        strategy_kwargs = {"max_words": chunk_size}
    elif chunking_strategy == "semantic":
        strategy_kwargs = {
            "model_name": embedding_model,
            "threshold": semantic_threshold,
        }

    strategy = get_strategy(chunking_strategy, **strategy_kwargs)

    t0 = time.perf_counter()
    all_chunks: list[Chunk] = []
    for p in passages:
        text = normalize_text(p["text"])
        metadata = {
            "passage_id": p["passage_id"],
            "query_id": p["query_id"],
            "language": p["language"],
            "is_selected": p["is_selected"],
        }
        chunks = strategy.chunk(text, metadata)
        all_chunks.extend(chunks)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Chunked %d passages → %d chunks in %.1fs (strategy=%s)",
        len(passages), len(all_chunks), elapsed, strategy.name,
    )

    # ── Step 3: Generate embeddings ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Generating embeddings")
    logger.info("=" * 60)

    embedder = EmbeddingGenerator(
        model_name=embedding_model,
        normalize=normalize_embeddings,
    )
    chunk_texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, batch_size=embedding_batch_size)

    # ── Step 4: Build FAISS index ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building FAISS index")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    dim = embeddings.shape[1]

    # WHY IndexFlatIP instead of IndexFlatL2:
    # With L2-normalized embeddings, inner product = cosine similarity.
    # Higher score = more similar (intuitive).
    # Your original used IndexFlatL2 where LOWER = more similar (confusing).
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = output_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    elapsed = time.perf_counter() - t0
    logger.info(
        "Built FAISS IndexFlatIP (dim=%d, vectors=%d) in %.1fs → %s",
        dim, index.ntotal, elapsed, faiss_path,
    )

    # ── Step 5: Build BM25 index ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Building BM25 index")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    tokenized_corpus = [tokenize_for_bm25(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = output_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Also save tokenized corpus for inspection
    bm25_corpus_path = output_dir / "bm25_corpus.json"
    with open(bm25_corpus_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    logger.info("Built BM25 index in %.1fs → %s", elapsed, bm25_path)

    # ── Step 6: Save chunk metadata ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Saving chunk metadata")
    logger.info("=" * 60)

    chunks_data = [
        {
            "text": c.text,
            "chunk_id": c.chunk_id,
            "passage_id": c.passage_id,
            "query_id": c.query_id if isinstance(c.query_id, int) else str(c.query_id),
            "chunk_index": c.chunk_index,
            "chunk_strategy": c.chunk_strategy,
            "language": c.language,
            "is_selected": c.is_selected,
        }
        for c in all_chunks
    ]

    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=None)
    
    chunks_size_mb = chunks_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %d chunks (%.1f MB) → %s", len(chunks_data), chunks_size_mb, chunks_path)

    # ── Step 7: Save queries (for evaluation) ───────────────────
    queries_path = output_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=None)
    logger.info("Saved %d queries → %s", len(queries), queries_path)

    # ── Step 8: Save embeddings ─────────────────────────────────
    embeddings_path = output_dir / "embeddings.npy"
    EmbeddingGenerator.save_embeddings(embeddings, embeddings_path)

    # ── Summary ─────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_t0
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total time:       %.1fs", total_elapsed)
    logger.info("  Passages loaded:  %d", len(passages))
    logger.info("  Chunks created:   %d", len(all_chunks))
    logger.info("  Embeddings:       %s", embeddings.shape)
    logger.info("  FAISS vectors:    %d", index.ntotal)
    logger.info("  Strategy:         %s", strategy.name)
    logger.info("  Output dir:       %s", output_dir)
    logger.info("=" * 60)

    return {
        "passages": len(passages),
        "chunks": len(all_chunks),
        "embeddings_shape": embeddings.shape,
        "faiss_vectors": index.ntotal,
        "strategy": strategy.name,
        "output_dir": str(output_dir),
        "total_seconds": total_elapsed,
    }
