"""
FastAPI Application Entry Point

PURPOSE:
    Initializes FastAPI, preloads all models and indexes at startup,
    and mounts API routes.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router, set_pipeline_instance
from config.settings import settings
from pipeline.graph import RAGPipelineGraph
from voice.stt import get_stt_client

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models and indexes on application startup."""
    logger.info("Initializing RAG Pipeline on server startup...")
    pipeline = RAGPipelineGraph(indexes_dir=settings.indexes_dir)
    stt = get_stt_client()
    set_pipeline_instance(pipeline, stt)
    logger.info("=" * 60)
    logger.info("  🚀 RAG Pipeline server is ready!")
    logger.info("  👉 Open in your browser: http://127.0.0.1:%d", settings.port)
    logger.info("  👉 Interactive Docs:     http://127.0.0.1:%d/docs", settings.port)
    logger.info("=" * 60)
    yield
    logger.info("Shutting down RAG service.")


app = FastAPI(
    title="Goa Hackathon 2026 — Voice-Enabled RAG",
    description="Production-grade, low-latency, grounded RAG system with Sarvam AI STT & LPU Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
