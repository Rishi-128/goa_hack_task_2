"""
Central configuration for the RAG pipeline.

WHY: Every magic number in the codebase lives here. This means:
  1. You can tune parameters without grep-searching the whole project.
  2. All values are overridable via environment variables or .env file.
  3. Pydantic validates types at startup — no runtime surprises.

USAGE:
  from config.settings import settings
  model = SentenceTransformer(settings.embedding_model)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with a fallback default."""
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration object. All values come from environment
    variables with sensible defaults.

    frozen=True prevents accidental mutation during a run.
    """

    # ── Project paths ───────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def indexes_dir(self) -> Path:
        return self.project_root / "indexes"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    # ── API Keys ────────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))

    # ── STT Provider ────────────────────────────────────────────────
    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "sarvam"))
    sarvam_language_code: str = field(default_factory=lambda: _env("SARVAM_LANGUAGE_CODE", "en-IN"))

    # ── Dataset ─────────────────────────────────────────────────────
    dataset_name: str = field(default_factory=lambda: _env("DATASET_NAME", "ai4bharat/MSMARCO-XI"))
    dataset_split: str = field(default_factory=lambda: _env("DATASET_SPLIT", "validation"))
    dataset_languages: str = field(default_factory=lambda: _env("DATASET_LANGUAGES", "eng_Latn"))
    dataset_sample_size: int = field(default_factory=lambda: _env_int("DATASET_SAMPLE_SIZE", 10000))

    # ── Embedding ───────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 256))
    normalize_embeddings: bool = field(
        default_factory=lambda: _env_bool("NORMALIZE_EMBEDDINGS", True)
    )

    # ── Chunking ────────────────────────────────────────────────────
    chunking_strategy: str = field(
        default_factory=lambda: _env("CHUNKING_STRATEGY", "passthrough")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    semantic_threshold: float = field(
        default_factory=lambda: _env_float("SEMANTIC_THRESHOLD", 0.5)
    )

    # ── Retrieval (Top-K Settings) ──────────────────────────────────
    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 15))
    sparse_top_k: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 15))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 10))
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 3))

    # ── Reranker & Fast Mode (<200ms) ───────────────────────────────
    # Set to false for ultra-low latency direct RRF ranking (<15ms)
    enable_reranker: bool = field(default_factory=lambda: _env_bool("ENABLE_RERANKER", False))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    reranker_max_length: int = field(default_factory=lambda: _env_int("RERANKER_MAX_LENGTH", 512))

    # ── LLM (Groq LPUs) ─────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-120b")
    )
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 350))

    # ── Query Processing ────────────────────────────────────────────
    default_query_mode: str = field(
        default_factory=lambda: _env("DEFAULT_QUERY_MODE", "normal")
    )
    auto_rewrite: bool = field(default_factory=lambda: _env_bool("AUTO_REWRITE", True))

    # ── Guardrails ──────────────────────────────────────────────────
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", -5.0)
    )
    offtopic_threshold: float = field(
        default_factory=lambda: _env_float("OFFTOPIC_THRESHOLD", 0.15)
    )
    grounding_threshold: float = field(
        default_factory=lambda: _env_float("GROUNDING_THRESHOLD", 0.3)
    )
    max_grounding_retries: int = field(
        default_factory=lambda: _env_int("MAX_GROUNDING_RETRIES", 1)
    )

    # ── Caching ─────────────────────────────────────────────────────
    enable_cache: bool = field(default_factory=lambda: _env_bool("ENABLE_CACHE", True))
    cache_max_size: int = field(default_factory=lambda: _env_int("CACHE_MAX_SIZE", 1000))

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ── Server ──────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))


# Singleton instance — import this everywhere
settings = Settings()
