"""
MSMARCO-XI Dataset Loader

PURPOSE:
    Load the MSMARCO-XI dataset from HuggingFace, extract passages,
    and produce a flat list of (passage_text, metadata) pairs ready
    for chunking and embedding.

WHY THIS EXISTS:
    Your original code loaded from local .txt files:
        DirectoryLoader("./agent/data", glob='*.txt')
    
    MSMARCO-XI is a HuggingFace dataset with a very different structure:
    each row has ~10 passages, language metadata, and ground truth labels.
    We need a dedicated loader that understands this structure.

ARCHITECTURE:
    HuggingFace streaming → filter by language → extract passages → 
    attach metadata → return List[dict]
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def load_msmarco_passages(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "validation",
    target_languages: Optional[list[str]] = None,
    sample_size: int = 10000,
    use_english_passages: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Load passages and queries from MSMARCO-XI.

    Returns:
        passages: List of dicts, each with:
            - text: passage text
            - passage_id: unique ID (f"{query_id}_{passage_index}")
            - query_id: the originating query ID
            - is_selected: 1 if this passage is ground-truth relevant
            - language: source or target language
        
        queries: List of dicts, each with:
            - query_id: int
            - query: str (English query)
            - answer: str (English answer, may be "No Answer Present.")
            - query_type: str
            - relevant_passage_ids: list of passage_ids marked is_selected=1
    """
    from datasets import load_dataset

    t0 = time.perf_counter()
    logger.info(
        "Loading dataset=%s split=%s languages=%s sample=%d",
        dataset_name, split, target_languages, sample_size,
    )

    ds = load_dataset(dataset_name, split=split, streaming=True)

    passages = []
    queries = []
    seen_passage_texts = set()  # deduplicate identical passages across queries
    passage_id_counter = 0
    rows_loaded = 0

    for row in ds:
        # Filter by target language if specified
        if target_languages:
            tl = row.get("target_lang", "")
            if tl not in target_languages:
                continue

        query_id = row.get("query_id")
        eng_query = row.get("Eng_Query", "")
        eng_answer = row.get("Eng_Answer", "")
        query_type = row.get("query_type", "")

        passages_data = row.get("passages", {})
        eng_passages = passages_data.get("English_passages", [])
        trans_passages = passages_data.get("Translated_passages", [])
        is_selected_list = passages_data.get("is_selected", [])

        # Choose which passages to use
        source_passages = eng_passages if use_english_passages else trans_passages

        relevant_passage_ids = []

        for idx, (passage_text, is_sel) in enumerate(
            zip(source_passages, is_selected_list)
        ):
            if not passage_text or len(passage_text.strip()) < 5:
                continue

            passage_text = passage_text.strip()

            # Deduplicate: same passage can appear across different queries
            # Use a hash to check without storing all texts in memory
            text_hash = hash(passage_text)
            if text_hash in seen_passage_texts:
                # Still track the passage_id for this query's ground truth
                # but don't add a duplicate passage to the corpus
                pid = f"{query_id}_{idx}"
                if is_sel == 1:
                    relevant_passage_ids.append(pid)
                continue

            seen_passage_texts.add(text_hash)
            pid = f"{query_id}_{idx}"

            passages.append({
                "text": passage_text,
                "passage_id": pid,
                "query_id": query_id,
                "passage_index": idx,
                "is_selected": is_sel,
                "language": "eng" if use_english_passages else row.get("target_lang", ""),
            })

            if is_sel == 1:
                relevant_passage_ids.append(pid)

            passage_id_counter += 1

        queries.append({
            "query_id": query_id,
            "query": eng_query,
            "answer": eng_answer,
            "query_type": query_type,
            "relevant_passage_ids": relevant_passage_ids,
        })

        rows_loaded += 1
        if sample_size > 0 and rows_loaded >= sample_size:
            break

    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %d rows → %d unique passages, %d queries in %.1fs",
        rows_loaded, len(passages), len(queries), elapsed,
    )

    # Log ground truth stats
    queries_with_answer = sum(
        1 for q in queries
        if q["answer"] and "No Answer Present" not in q["answer"]
    )
    queries_with_relevant = sum(
        1 for q in queries if q["relevant_passage_ids"]
    )
    logger.info(
        "Queries with answer: %d/%d (%.1f%%), with relevant passages: %d/%d (%.1f%%)",
        queries_with_answer, len(queries),
        100 * queries_with_answer / max(len(queries), 1),
        queries_with_relevant, len(queries),
        100 * queries_with_relevant / max(len(queries), 1),
    )

    return passages, queries
