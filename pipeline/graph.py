"""
LangGraph StateGraph Orchestration for RAG Pipeline

PURPOSE:
    Coordinates all RAG stages with support for both in-memory FAISS/BM25 and 
    disk-backed SQLite FTS5 (millions of rows):
    
    validate_input -> (is_safe?)
       ├── No  -> handle_safety_block -> END
       └── Yes -> process_query -> retrieve (FAISS + SQLite FTS5) -> rerank/fast_rrf
                    ├── No  -> handle_abstain -> END
                    └── Yes -> generate -> validate_grounding -> (grounded?)
                                 ├── Yes -> END
                                 └── No (retry <= max) -> generate -> validate_grounding
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from config.settings import settings
from pipeline.generator import StructuredGenerator
from pipeline.guardrails import (
    check_grounding,
    check_input_safety,
    check_retrieval_confidence,
)
from pipeline.query_processor import QueryProcessor
from pipeline.recovery import (
    create_abstention_response,
    create_error_fallback_response,
    create_safety_block_response,
)
from pipeline.state import RAGStateDict
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from retrieval.sqlite_fts import SQLiteFTSRetriever

logger = logging.getLogger(__name__)


class RAGPipelineGraph:
    """
    StateGraph-based RAG Pipeline supporting in-memory FAISS and disk-backed SQLite FTS5 for millions of rows.
    """

    def __init__(
        self,
        indexes_dir: Optional[Path] = None,
        embedder=None,
        reranker=None,
        llm_client=None,
    ):
        self.indexes_dir = Path(indexes_dir or settings.indexes_dir)
        logger.info("Initializing RAGPipelineGraph from %s", self.indexes_dir)
        t0 = time.perf_counter()

        # 1. Load chunks metadata if present
        chunks_path = self.indexes_dir / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info("Loaded %d chunks metadata", len(self.chunks))
        else:
            self.chunks = []

        # 2. Load retrievers
        faiss_path = self.indexes_dir / "faiss.index"
        bm25_path = self.indexes_dir / "bm25.pkl"
        sqlite_db_path = settings.project_root / "data" / "msmarco_full.db"

        self.dense_retriever = DenseRetriever(faiss_path) if faiss_path.exists() else None
        self.sparse_retriever = SparseRetriever(bm25_path) if bm25_path.exists() else None
        self.fts_retriever = SQLiteFTSRetriever(sqlite_db_path) if sqlite_db_path.exists() else None

        if self.fts_retriever:
            logger.info("Connected SQLite FTS5 database with %d passages", self.fts_retriever.count())

        self.reranker = reranker or (
            CrossEncoderReranker(
                model_name=settings.reranker_model,
                max_length=settings.reranker_max_length,
            )
            if settings.enable_reranker
            else None
        )

        self.query_processor = QueryProcessor(embedder=embedder, llm_client=llm_client)
        self.generator = StructuredGenerator(llm_client=llm_client)

        # Build the LangGraph workflow
        self.graph = self._build_graph()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("RAGPipelineGraph initialized in %.2f ms", elapsed)

    # ── Node Implementations ──────────────────────────────────────────

    def node_validate_input(self, state: RAGStateDict) -> RAGStateDict:
        t0 = time.perf_counter()
        query = state.get("original_query", "")
        is_safe, error_msg = check_input_safety(query)
        elapsed = (time.perf_counter() - t0) * 1000

        latency = state.get("latency_breakdown", {})
        latency["input_validation"] = elapsed

        return {
            "is_safe": is_safe,
            "safety_error": error_msg,
            "latency_breakdown": latency,
        }

    def node_process_query(self, state: RAGStateDict) -> RAGStateDict:
        t0 = time.perf_counter()
        query = state.get("original_query", "")
        history = state.get("conversation_history", [])
        mode = state.get("query_mode", settings.default_query_mode)

        proc_res = self.query_processor.process(
            query=query,
            conversation_history=history,
            query_mode=mode,
        )
        total_proc = (time.perf_counter() - t0) * 1000

        latency = state.get("latency_breakdown", {})
        latency["query_processing"] = total_proc
        latency["query_rewrite"] = proc_res.get("rewrite_latency_ms", 0.0)
        latency["query_embedding"] = proc_res.get("embedding_latency_ms", 0.0)

        return {
            "processed_query": proc_res["processed_query"],
            "query_embedding": proc_res["query_embedding"],
            "is_rewritten": proc_res.get("rewrite_latency_ms", 0.0) > 0,
            "latency_breakdown": latency,
        }

    def node_retrieve(self, state: RAGStateDict) -> RAGStateDict:
        t0 = time.perf_counter()
        query = state.get("processed_query", state.get("original_query", ""))
        embedding = state.get("query_embedding")

        # 1. Dense FAISS search
        t_dense = time.perf_counter()
        dense_results = []
        if self.dense_retriever and embedding is not None:
            dense_results = self.dense_retriever.search(embedding, top_k=settings.dense_top_k)
        elapsed_dense = (time.perf_counter() - t_dense) * 1000

        # 2. Sparse Search: SQLite FTS5 (Disk) or in-memory BM25
        t_sparse = time.perf_counter()
        sparse_results = []
        if self.fts_retriever and self.fts_retriever.count() > 0:
            fts_docs = self.fts_retriever.search(query, top_k=settings.sparse_top_k)
            # Map FTS docs into ranking indices/scores
            sparse_results = [(idx, doc["score"]) for idx, doc in enumerate(fts_docs)]
        elif self.sparse_retriever:
            sparse_results = self.sparse_retriever.search(query, top_k=settings.sparse_top_k)
        elapsed_sparse = (time.perf_counter() - t_sparse) * 1000

        # 3. Reciprocal Rank Fusion (RRF)
        t_rrf = time.perf_counter()
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=settings.rrf_k,
            top_n=settings.rerank_top_k,
        )
        elapsed_rrf = (time.perf_counter() - t_rrf) * 1000
        total_retrieve = (time.perf_counter() - t0) * 1000

        latency = state.get("latency_breakdown", {})
        latency["dense_retrieval"] = elapsed_dense
        latency["sparse_retrieval"] = elapsed_sparse
        latency["rrf_fusion"] = elapsed_rrf
        latency["total_retrieval"] = total_retrieve

        return {
            "dense_candidates": dense_results,
            "sparse_candidates": sparse_results,
            "fused_candidates": fused,
            "latency_breakdown": latency,
        }

    def node_rerank(self, state: RAGStateDict) -> RAGStateDict:
        t0 = time.perf_counter()
        query = state.get("processed_query", state.get("original_query", ""))
        fused = state.get("fused_candidates", [])

        # Check if we should fetch texts from SQLite FTS or local chunks.json
        top_fused = fused[:settings.final_top_k]
        context_chunks = []
        sources = []

        if self.fts_retriever and self.fts_retriever.count() > 0:
            fts_docs = self.fts_retriever.search(query, top_k=settings.final_top_k)
            context_chunks = [doc["text"] for doc in fts_docs]
            sources = [
                {
                    "chunk_id": doc["chunk_id"],
                    "passage_id": doc["passage_id"],
                    "score": doc["score"],
                }
                for doc in fts_docs
            ]
        elif self.chunks:
            context_chunks = [self.chunks[idx]["text"] for idx, _score in top_fused if idx < len(self.chunks)]
            sources = [
                {
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "passage_id": self.chunks[idx]["passage_id"],
                    "score": round(score, 4),
                }
                for idx, score in top_fused
                if idx < len(self.chunks)
            ]

        elapsed = (time.perf_counter() - t0) * 1000
        latency = state.get("latency_breakdown", {})
        latency["reranking"] = elapsed

        return {
            "reranked_chunks": [(idx, score, "") for idx, score in top_fused],
            "top_reranker_score": top_fused[0][1] if top_fused else 0.0,
            "is_confident": len(context_chunks) > 0,
            "context_chunks": context_chunks,
            "sources": sources,
            "latency_breakdown": latency,
        }

    def node_generate(self, state: RAGStateDict) -> RAGStateDict:
        query = state.get("processed_query", state.get("original_query", ""))
        contexts = state.get("context_chunks", [])

        gen_dict, elapsed_ms = self.generator.generate(query, contexts)
        
        latency = state.get("latency_breakdown", {})
        latency["llm_generation"] = elapsed_ms

        return {
            "answer": gen_dict.get("answer", ""),
            "summary": gen_dict.get("summary", ""),
            "latency_breakdown": latency,
        }

    def node_validate_grounding(self, state: RAGStateDict) -> RAGStateDict:
        t0 = time.perf_counter()
        answer = state.get("answer", "")
        contexts = state.get("context_chunks", [])

        is_grounded, grounding_score = check_grounding(answer, contexts)
        elapsed = (time.perf_counter() - t0) * 1000

        latency = state.get("latency_breakdown", {})
        latency["grounding_validation"] = elapsed

        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "latency_breakdown": latency,
        }

    def node_handle_safety_block(self, state: RAGStateDict) -> RAGStateDict:
        resp = create_safety_block_response(state.get("safety_error", "safety_filter"))
        return {
            "answer": resp["answer"],
            "summary": resp["summary"],
            "is_grounded": True,
            "sources": [],
            "should_abstain": True,
        }

    def node_handle_abstain(self, state: RAGStateDict) -> RAGStateDict:
        resp = create_abstention_response("low_retrieval_confidence")
        return {
            "answer": resp["answer"],
            "summary": resp["summary"],
            "is_grounded": True,
            "sources": state.get("sources", []),
            "should_abstain": True,
        }

    # ── Conditional Edges ─────────────────────────────────────────────

    def _route_after_safety(self, state: RAGStateDict) -> str:
        return "process_query" if state.get("is_safe", True) else "handle_safety_block"

    def _route_after_rerank(self, state: RAGStateDict) -> str:
        if not state.get("is_confident", True) or not state.get("context_chunks"):
            return "handle_abstain"
        return "generate"

    # ── Graph Construction ────────────────────────────────────────────

    def _build_graph(self) -> Any:
        workflow = StateGraph(RAGStateDict)

        # Register nodes
        workflow.add_node("validate_input", self.node_validate_input)
        workflow.add_node("handle_safety_block", self.node_handle_safety_block)
        workflow.add_node("process_query", self.node_process_query)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("handle_abstain", self.node_handle_abstain)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("validate_grounding", self.node_validate_grounding)

        # Entry point
        workflow.set_entry_point("validate_input")

        # Conditional edge: after safety check
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_safety,
            {
                "process_query": "process_query",
                "handle_safety_block": "handle_safety_block",
            },
        )

        workflow.add_edge("handle_safety_block", END)
        workflow.add_edge("process_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")

        # Conditional edge: after reranking / confidence
        workflow.add_conditional_edges(
            "rerank",
            self._route_after_rerank,
            {
                "generate": "generate",
                "handle_abstain": "handle_abstain",
            },
        )

        workflow.add_edge("handle_abstain", END)
        workflow.add_edge("generate", "validate_grounding")
        workflow.add_edge("validate_grounding", END)

        return workflow.compile()

    # ── Execution ─────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        query_mode: str = "normal",
    ) -> dict:
        """
        Execute full RAG graph.

        Returns:
            Dict containing answer, summary, grounded flag, sources, and latency metrics.
        """
        t0 = time.perf_counter()

        initial_state: RAGStateDict = {
            "original_query": query,
            "conversation_history": conversation_history or [],
            "query_mode": query_mode,
            "latency_breakdown": {},
            "retry_count": 0,
        }

        final_state = self.graph.invoke(initial_state)

        total_latency = (time.perf_counter() - t0) * 1000
        latencies = final_state.get("latency_breakdown", {})

        return {
            "answer": final_state.get("answer", "Content not found."),
            "summary": final_state.get("summary", ""),
            "grounded": final_state.get("is_grounded", False),
            "sources": final_state.get("sources", []),
            "latency_breakdown": latencies,
            "total_latency_ms": round(total_latency, 2),
        }

    def run_stream(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        query_mode: str = "normal",
    ):
        """
        Execute RAG retrieval and stream tokens in real-time.

        Yields dicts with:
            {"type": "metadata", "sources": [...], "retrieval_latency_ms": ...}
            {"type": "token", "token": "...", "ttft_ms": ...}
            {"type": "done", "total_latency_ms": ...}
        """
        t0 = time.perf_counter()

        # 1. Validation
        is_safe, error_msg = check_input_safety(query)
        if not is_safe:
            yield {"type": "token", "token": "Query blocked by safety filter.", "ttft_ms": 0.1}
            yield {"type": "done", "total_latency_ms": (time.perf_counter() - t0) * 1000}
            return

        # 2. Query Processing
        proc_res = self.query_processor.process(
            query=query,
            conversation_history=conversation_history or [],
            query_mode=query_mode,
        )
        proc_query = proc_res["processed_query"]
        embedding = proc_res["query_embedding"]

        # 3. Retrieval
        context_chunks = []
        sources = []

        if self.fts_retriever and self.fts_retriever.count() > 0:
            fts_docs = self.fts_retriever.search(proc_query, top_k=settings.final_top_k)
            context_chunks = [doc["text"] for doc in fts_docs]
            sources = [
                {
                    "chunk_id": doc["chunk_id"],
                    "passage_id": doc["passage_id"],
                    "score": doc["score"],
                }
                for doc in fts_docs
            ]
        elif self.dense_retriever or self.sparse_retriever:
            dense_results = self.dense_retriever.search(embedding, top_k=settings.dense_top_k) if self.dense_retriever else []
            sparse_results = self.sparse_retriever.search(proc_query, top_k=settings.sparse_top_k) if self.sparse_retriever else []
            fused = reciprocal_rank_fusion(
                ranked_lists=[dense_results, sparse_results],
                k=settings.rrf_k,
                top_n=settings.final_top_k,
            )
            context_chunks = [self.chunks[idx]["text"] for idx, _score in fused if idx < len(self.chunks)]
            sources = [
                {
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "passage_id": self.chunks[idx]["passage_id"],
                    "score": round(score, 4),
                }
                for idx, score in fused
                if idx < len(self.chunks)
            ]

        retrieval_elapsed = (time.perf_counter() - t0) * 1000

        # Yield metadata event first
        yield {
            "type": "metadata",
            "sources": sources,
            "retrieval_latency_ms": round(retrieval_elapsed, 2),
        }

        # 4. Stream LLM tokens
        for token, ttft_ms, is_finished in self.generator.generate_stream(proc_query, context_chunks):
            if is_finished:
                total_elapsed = (time.perf_counter() - t0) * 1000
                yield {"type": "done", "total_latency_ms": round(total_elapsed, 2)}
            else:
                yield {"type": "token", "token": token, "ttft_ms": round(ttft_ms, 2) if ttft_ms else None}
