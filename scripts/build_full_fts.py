"""
High-Speed Streaming Ingestion for Full 55GB MSMARCO-XI Dataset into SQLite FTS5

PURPOSE:
    Streams English and Indic passages from ai4bharat/MSMARCO-XI and inserts
    into SQLite FTS5 table with live progress updates every 2 seconds.

USAGE:
    python scripts/build_full_fts.py
    python scripts/build_full_fts.py --split train
    python scripts/build_full_fts.py --max-passages 500000
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.sqlite_fts import SQLiteFTSRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest_msmarco_stream(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "validation",
    batch_size: int = 2000,
    max_passages: int = 0,  # 0 = unlimited / full split
    target_languages: list[str] = None,
    use_both_eng_and_translated: bool = True,
):
    """
    Stream rows from HuggingFace and insert into SQLite FTS5 in bulk transactions.
    """
    from datasets import load_dataset

    retriever = SQLiteFTSRetriever()
    initial_count = retriever.count()
    print(f"🚀 SQLite FTS5 database initialized at: {retriever.db_path}", flush=True)
    print(f"📦 Existing passages in DB: {initial_count:,}", flush=True)
    print(f"🌐 Connecting to HuggingFace dataset: {dataset_name} (split: '{split}', streaming: True)...", flush=True)

    ds = load_dataset(dataset_name, split=split, streaming=True)

    batch = []
    total_inserted = 0
    t0 = time.perf_counter()
    last_log_time = t0
    seen_texts = set()

    print(f"⚡ Streaming and indexing passages into SQLite FTS5... (Press Ctrl+C to pause/stop anytime)\n", flush=True)

    try:
        for row_idx, row in enumerate(ds):
            # Optional language filter
            if target_languages:
                tl = row.get("target_lang", row.get("lang", ""))
                if tl and tl not in target_languages:
                    continue

            query_id = row.get("query_id", row_idx)
            passages_data = row.get("passages", {})

            # MSMARCO-XI uses 'English_passages' and 'Translated_passages'
            eng_passages = passages_data.get("English_passages", [])
            trans_passages = passages_data.get("Translated_passages", [])
            fallback_passages = passages_data.get("passage_text", [])

            # Collect candidate passages for this row
            collected_passages = []
            if eng_passages:
                collected_passages.extend(eng_passages)
            if use_both_eng_and_translated and trans_passages:
                collected_passages.extend(trans_passages)
            if not collected_passages and fallback_passages:
                collected_passages.extend(fallback_passages)

            for p_idx, text in enumerate(collected_passages):
                if not text or not isinstance(text, str):
                    continue
                clean_text = " ".join(text.split()).strip()
                if len(clean_text) < 15:
                    continue

                # Memory-safe batch deduplication
                h = hash(clean_text)
                if h in seen_texts:
                    continue
                if len(seen_texts) > 250000:
                    seen_texts.clear()
                seen_texts.add(h)

                pid = f"{query_id}_{p_idx}"
                batch.append((pid, clean_text))

                if len(batch) >= batch_size:
                    retriever.insert_batch(batch)
                    total_inserted += len(batch)
                    batch = []

                    now = time.perf_counter()
                    if now - last_log_time >= 2.0:
                        elapsed = now - t0
                        speed = total_inserted / elapsed if elapsed > 0 else 0
                        db_size_mb = os.path.getsize(retriever.db_path) / (1024 * 1024)
                        total_current = total_inserted + initial_count
                        print(
                            f"⚡ Indexed: {total_current:,} passages | Speed: {speed:,.1f} passages/sec | DB Size: {db_size_mb:.1f} MB",
                            flush=True,
                        )
                        last_log_time = now

                if max_passages > 0 and (total_inserted + initial_count) >= max_passages:
                    break

            if max_passages > 0 and (total_inserted + initial_count) >= max_passages:
                break

        # Flush any remaining rows
        if batch:
            retriever.insert_batch(batch)
            total_inserted += len(batch)

    except KeyboardInterrupt:
        print("\n🛑 Ingestion paused by user. Saving final batch to disk...", flush=True)
        if batch:
            retriever.insert_batch(batch)
            total_inserted += len(batch)

    total_time = time.perf_counter() - t0
    final_count = retriever.count()
    db_size_mb = os.path.getsize(retriever.db_path) / (1024 * 1024)

    print("\n" + "=" * 65, flush=True)
    print("✅ INGESTION COMPLETE!", flush=True)
    print(f"   📊 Total Passages in DB: {final_count:,}", flush=True)
    print(f"   📥 Passages Ingested:    {total_inserted:,}", flush=True)
    print(f"   ⏱️ Time Taken:          {total_time:.2f} seconds", flush=True)
    print(f"   💾 Final DB File Size:   {db_size_mb:.2f} MB ({retriever.db_path})", flush=True)
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream MSMARCO-XI into SQLite FTS5")
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XI", help="HuggingFace dataset name")
    parser.add_argument("--split", default="validation", help="Dataset split: 'validation' or 'train'")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch insertion size")
    parser.add_argument("--max-passages", type=int, default=0, help="Max passages to index (0 = unlimited)")
    args = parser.parse_args()

    ingest_msmarco_stream(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_passages=args.max_passages,
    )
