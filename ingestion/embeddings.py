"""
Batch Embedding Generation

PURPOSE:
    Generate embeddings for passages in batches and return as numpy array.
    
WHY SEPARATE FROM INDEX BUILDER:
    Embedding generation is the most expensive offline step.
    Keeping it separate lets us:
    1. Cache embeddings to disk
    2. Swap embedding models without rebuilding everything
    3. Benchmark different models independently
    
DESIGN DECISION — Normalization:
    We L2-normalize all embeddings. This means:
    - Cosine similarity = dot product (faster)
    - We can use FAISS IndexFlatIP instead of IndexFlatL2
    - Your original code used IndexFlatL2 which computes L2 distance,
      but MiniLM produces unit-norm embeddings, so cosine is more appropriate.
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    
    Wraps the model to provide:
    - Batch processing with progress logging
    - L2 normalization
    - Save/load to disk
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize

        logger.info("Loading embedding model: %s", model_name)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        elapsed = time.perf_counter() - t0
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded in %.1fs (dim=%d)", elapsed, self.dimension
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of passage texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), dimension), dtype float32
        """
        logger.info("Encoding %d texts (batch_size=%d)", len(texts), batch_size)
        t0 = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        elapsed = time.perf_counter() - t0
        rate = len(texts) / max(elapsed, 0.001)
        logger.info(
            "Encoded %d texts in %.1fs (%.0f texts/sec), shape=%s",
            len(texts), elapsed, rate, embeddings.shape,
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query. Used at inference time.
        Returns a 2D array of shape (1, dimension) for FAISS compatibility.
        """
        emb = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.array(emb, dtype=np.float32)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Path):
        """Save embeddings to disk as .npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Saved embeddings to %s (%.1f MB)", path, size_mb)

    @staticmethod
    def load_embeddings(path: Path) -> np.ndarray:
        """Load embeddings from disk."""
        path = Path(path)
        embeddings = np.load(path)
        logger.info("Loaded embeddings from %s, shape=%s", path, embeddings.shape)
        return embeddings
