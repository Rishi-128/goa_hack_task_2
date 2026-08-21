"""
Dataset Analysis Script for MSMARCO-XI

PURPOSE:
    Inspect the MSMARCO-XI dataset before building the ingestion pipeline.
    Never assume dataset structure — verify it.

WHAT IT REPORTS:
    - Total records and splits
    - Column names and types
    - Language distribution (source_lang, target_lang)
    - Query type distribution
    - Passage count per row
    - Passage length statistics (mean, median, max, min, percentiles)
    - is_selected distribution (how many passages are marked relevant)
    - Missing/null values
    - Duplicate query_id rates
    - "No Answer Present" rate
    - Sample rows for visual inspection

USAGE:
    python -m scripts.analyze_dataset
    python -m scripts.analyze_dataset --split validation --sample 5000
"""

import argparse
import io
import sys
import time
from collections import Counter
from pathlib import Path

# Fix Windows console encoding — cp1252 can't handle Indic scripts
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_dataset_sample(dataset_name: str, split: str, sample_size: int):
    """Load a sample from the HuggingFace dataset."""
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"  Loading: {dataset_name}")
    print(f"  Split:   {split}")
    print(f"  Sample:  {sample_size if sample_size > 0 else 'ALL (WARNING: very large)'}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    # Stream to avoid downloading the full 55GB dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)

    # Take a sample
    if sample_size > 0:
        rows = []
        for i, row in enumerate(ds):
            if i >= sample_size:
                break
            rows.append(row)
    else:
        # WARNING: This will try to load everything
        rows = list(ds.take(100000))  # Safety cap

    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(rows)} rows in {elapsed:.1f}s\n")
    return rows


def analyze_schema(rows):
    """Report column names and types."""
    print(f"\n{'='*60}")
    print("  SCHEMA")
    print(f"{'='*60}")

    if not rows:
        print("  No rows loaded!")
        return

    sample = rows[0]
    for key, value in sample.items():
        vtype = type(value).__name__
        if isinstance(value, dict):
            subkeys = list(value.keys())
            print(f"  {key:25s}  {vtype:10s}  sub-keys: {subkeys}")
        elif isinstance(value, list):
            inner = type(value[0]).__name__ if value else "empty"
            print(f"  {key:25s}  {vtype:10s}  [{inner}] len={len(value)}")
        else:
            preview = str(value)[:80]
            # Safely handle non-ASCII characters for display
            safe_preview = preview.encode('ascii', errors='replace').decode('ascii')
            print(f"  {key:25s}  {vtype:10s}  example: {safe_preview}")


def analyze_languages(rows):
    """Report language distribution."""
    print(f"\n{'='*60}")
    print("  LANGUAGE DISTRIBUTION")
    print(f"{'='*60}")

    source_langs = Counter(r.get("source_lang", "MISSING") for r in rows)
    target_langs = Counter(r.get("target_lang", "MISSING") for r in rows)

    print("\n  Source Languages:")
    for lang, count in source_langs.most_common():
        pct = 100 * count / len(rows)
        print(f"    {lang:20s}  {count:>8,d}  ({pct:5.1f}%)")

    print("\n  Target Languages:")
    for lang, count in target_langs.most_common():
        pct = 100 * count / len(rows)
        print(f"    {lang:20s}  {count:>8,d}  ({pct:5.1f}%)")


def analyze_query_types(rows):
    """Report query type distribution."""
    print(f"\n{'='*60}")
    print("  QUERY TYPE DISTRIBUTION")
    print(f"{'='*60}")

    qtypes = Counter(r.get("query_type", "MISSING") for r in rows)
    for qtype, count in qtypes.most_common():
        pct = 100 * count / len(rows)
        print(f"    {qtype:20s}  {count:>8,d}  ({pct:5.1f}%)")


def analyze_passages(rows):
    """Report passage statistics."""
    print(f"\n{'='*60}")
    print("  PASSAGE STATISTICS")
    print(f"{'='*60}")

    eng_passage_counts = []
    trans_passage_counts = []
    eng_passage_lengths = []  # in characters
    trans_passage_lengths = []
    is_selected_counts = []  # how many passages are selected per row
    total_selected = 0
    total_passages = 0

    for r in rows:
        passages = r.get("passages", {})

        eng = passages.get("English_passages", [])
        trans = passages.get("Translated_passages", [])
        selected = passages.get("is_selected", [])

        eng_passage_counts.append(len(eng))
        trans_passage_counts.append(len(trans))

        for p in eng:
            if p:
                eng_passage_lengths.append(len(p))
        for p in trans:
            if p:
                trans_passage_lengths.append(len(p))

        num_selected = sum(1 for s in selected if s == 1)
        is_selected_counts.append(num_selected)
        total_selected += num_selected
        total_passages += len(selected)

    import statistics

    def _stats(values, label):
        if not values:
            print(f"\n  {label}: No data")
            return
        values_sorted = sorted(values)
        n = len(values_sorted)
        print(f"\n  {label}:")
        print(f"    Count:   {n:>10,d}")
        print(f"    Mean:    {statistics.mean(values):>10.1f}")
        print(f"    Median:  {statistics.median(values):>10.1f}")
        print(f"    Min:     {min(values):>10,d}")
        print(f"    Max:     {max(values):>10,d}")
        print(f"    P25:     {values_sorted[int(n * 0.25)]:>10,d}")
        print(f"    P75:     {values_sorted[int(n * 0.75)]:>10,d}")
        print(f"    P95:     {values_sorted[int(n * 0.95)]:>10,d}")

    _stats(eng_passage_counts, "English Passages Per Row")
    _stats(trans_passage_counts, "Translated Passages Per Row")
    _stats(eng_passage_lengths, "English Passage Length (chars)")
    _stats(trans_passage_lengths, "Translated Passage Length (chars)")

    print(f"\n  is_selected Distribution:")
    selected_dist = Counter(is_selected_counts)
    for num_sel, count in sorted(selected_dist.items()):
        pct = 100 * count / len(rows)
        print(f"    {num_sel} selected:  {count:>8,d}  ({pct:5.1f}%)")

    if total_passages > 0:
        print(f"\n  Total passages:         {total_passages:>10,d}")
        print(f"  Total selected:         {total_selected:>10,d}")
        print(f"  Selection rate:         {100*total_selected/total_passages:>9.2f}%")


def analyze_answers(rows):
    """Report answer statistics and 'No Answer Present' rate."""
    print(f"\n{'='*60}")
    print("  ANSWER STATISTICS")
    print(f"{'='*60}")

    no_answer_eng = 0
    no_answer_target = 0
    eng_answer_lengths = []
    target_answer_lengths = []

    for r in rows:
        eng_ans = r.get("Eng_Answer", "")
        target_ans = r.get("Answer", "")

        if eng_ans and "No Answer Present" in eng_ans:
            no_answer_eng += 1
        if eng_ans:
            eng_answer_lengths.append(len(eng_ans))

        if target_ans:
            target_answer_lengths.append(len(target_ans))
            # Check for translated "no answer" patterns
            if "No Answer Present" in target_ans or len(target_ans) < 5:
                no_answer_target += 1

    print(f"\n  'No Answer Present' in Eng_Answer:  {no_answer_eng:>8,d} / {len(rows):,d}  ({100*no_answer_eng/len(rows):.1f}%)")

    import statistics
    if eng_answer_lengths:
        print(f"\n  English Answer Length (chars):")
        print(f"    Mean:    {statistics.mean(eng_answer_lengths):>10.1f}")
        print(f"    Median:  {statistics.median(eng_answer_lengths):>10.1f}")
        print(f"    Max:     {max(eng_answer_lengths):>10,d}")


def analyze_duplicates(rows):
    """Report duplicate query_id rates."""
    print(f"\n{'='*60}")
    print("  DUPLICATES & MISSING VALUES")
    print(f"{'='*60}")

    query_ids = [r.get("query_id") for r in rows]
    unique_ids = set(query_ids)
    print(f"\n  Total rows:             {len(rows):>10,d}")
    print(f"  Unique query_ids:       {len(unique_ids):>10,d}")
    print(f"  Duplicate query_ids:    {len(rows) - len(unique_ids):>10,d}")

    # Duplicates are expected — same query_id appears across languages
    if len(rows) > len(unique_ids):
        # Check if duplicates are across languages
        id_langs = Counter()
        for r in rows:
            qid = r.get("query_id")
            lang = r.get("target_lang", "?")
            id_langs[(qid, lang)] += 1

        cross_lang_dups = sum(1 for v in id_langs.values() if v > 1)
        print(f"  Same (query_id, lang):  {cross_lang_dups:>10,d}  (true duplicates within same language)")

    # Missing values
    print(f"\n  Missing Values:")
    fields_to_check = ["source_lang", "target_lang", "query_id", "query_type",
                        "query", "Eng_Query", "Answer", "Eng_Answer"]
    for field in fields_to_check:
        missing = sum(1 for r in rows if not r.get(field))
        if missing > 0:
            print(f"    {field:25s}  {missing:>8,d} missing")
        else:
            print(f"    {field:25s}  ✓ complete")


def print_samples(rows, n=3):
    """Print a few sample rows for visual inspection."""
    print(f"\n{'='*60}")
    print(f"  SAMPLE ROWS (first {n})")
    print(f"{'='*60}")

    for i, r in enumerate(rows[:n]):
        print(f"\n  -- Row {i} --")
        print(f"  query_id:     {r.get('query_id')}")
        print(f"  query_type:   {r.get('query_type')}")
        print(f"  source_lang:  {r.get('source_lang')}")
        print(f"  target_lang:  {r.get('target_lang')}")
        print(f"  Eng_Query:    {r.get('Eng_Query', '')[:100]}")
        # Safely display non-ASCII query/answer fields
        query_safe = str(r.get('query', ''))[:100].encode('ascii', errors='replace').decode('ascii')
        print(f"  query:        {query_safe}")
        print(f"  Eng_Answer:   {r.get('Eng_Answer', '')[:100]}")
        answer_safe = str(r.get('Answer', ''))[:100].encode('ascii', errors='replace').decode('ascii')
        print(f"  Answer:       {answer_safe}")

        passages = r.get("passages", {})
        eng_p = passages.get("English_passages", [])
        sel = passages.get("is_selected", [])
        print(f"  passages:     {len(eng_p)} English passages")
        print(f"  is_selected:  {sel}")
        if eng_p:
            print(f"  passage[0]:   {eng_p[0][:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze MSMARCO-XI dataset")
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XI",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", default="validation",
                        help="Dataset split (train/validation)")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Number of rows to sample (0 for all)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MSMARCO-XI DATASET ANALYSIS")
    print("=" * 60)

    rows = load_dataset_sample(args.dataset, args.split, args.sample)

    if not rows:
        print("ERROR: No rows loaded. Check dataset name and network.")
        sys.exit(1)

    analyze_schema(rows)
    analyze_languages(rows)
    analyze_query_types(rows)
    analyze_passages(rows)
    analyze_answers(rows)
    analyze_duplicates(rows)
    print_samples(rows)

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n  KEY TAKEAWAYS FOR INGESTION:")
    print(f"  1. Each row has ~10 passages — these are our 'chunks'")
    print(f"  2. 'is_selected' gives ground truth for retrieval eval")
    print(f"  3. 'No Answer Present' rows test abstention capability")
    print(f"  4. Same query_id across languages = multilingual versions")
    print(f"  5. Use English passages for baseline, translated for multilingual demo")
    print()


if __name__ == "__main__":
    main()
