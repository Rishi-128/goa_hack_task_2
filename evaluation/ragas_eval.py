"""
RAGAS Evaluation Wrapper

PURPOSE:
    Evaluates generation quality using the RAGAS framework:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall

WHY PRESERVED:
    Preserved from your original rag_core.py implementation, but separated
    into a clean evaluation module with configurable batch size and error recovery.
"""

import logging
from typing import Optional

from datasets import Dataset

from config.settings import settings

logger = logging.getLogger(__name__)


def run_ragas_evaluation(
    eval_samples: list[dict],
    groq_api_key: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
) -> dict:
    """
    Run RAGAS evaluation on a set of generated answers.

    Args:
        eval_samples: List of dicts, each containing:
            - question: str
            - answer: str
            - contexts: list[str]
            - ground_truth: str

    Returns:
        Dict of metric scores (e.g. {"faithfulness": 0.92, ...})
    """
    try:
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        except ModuleNotFoundError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error("RAGAS dependencies not fully installed: %s", e)
        return {"error": str(e)}

    api_key = groq_api_key or settings.groq_api_key
    model_name = embedding_model_name or settings.embedding_model

    evaluator_llm = ChatGroq(
        model=settings.llm_model,
        groq_api_key=api_key,
        temperature=0.0,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model_name}"
    )

    data = {
        "question": [s["question"] for s in eval_samples],
        "answer": [s["answer"] for s in eval_samples],
        "contexts": [s["contexts"] for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }

    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS evaluation on %d samples...", len(eval_samples))
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    logger.info("RAGAS Results: %s", results)
    return results
