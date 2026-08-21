# Goa Hackathon 2026 — Task 2: Voice-Enabled RAG System

A production-quality, low-latency, grounded, voice-enabled Retrieval-Augmented Generation (RAG) system built for the MSMARCO-XI dataset and evaluated with IR and generation metrics.

---

## 1. Architectural Overview

```text
                                  USER
                                    │
                             ┌──────┴──────┐
                             │  🎤 Voice   │  📝 Text
                             │  Input      │  Input
                             └──────┬──────┘
                                    │
                             ┌──────┴──────┐
                             │  STT        │  (ElevenLabs)
                             └──────┬──────┘
                                    │
                             ┌──────┴──────┐
                             │  Safety     │── REJECT → Controlled safety message
                             │  Filter     │
                             └──────┬──────┘
                                    │
                             ┌──────┴──────┐
                             │ Conditional │
                             │ Rewrite     │── Standalone? Bypass LLM (0 ms overhead)
                             └──────┬──────┘   Context-dependent? Rewrite with LLM
                                    │
                             ┌──────┴──────┐
                             │ Embedding   │  (all-MiniLM-L6-v2, L2 Normalized)
                             └──────┬──────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐
                  │ FAISS Dense │       │ BM25 Sparse │
                  │ IndexFlatIP │       │ Corpus Match│
                  └──────┬──────┘       └──────┬──────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                             ┌─────────────┐
                             │ RRF Fusion  │  k=60 (Scale-invariant)
                             └──────┬──────┘
                                    ▼
                             ┌─────────────┐
                             │ CrossEncoder│  BAAI/bge-reranker-base (Batched)
                             │ Reranking   │
                             └──────┬──────┘
                                    ▼
                             ┌─────────────┐
                             │ Confidence  │── Score < Threshold → Controlled Abstention
                             │ Gate        │
                             └──────┬──────┘
                                    ▼
                             ┌─────────────┐
                             │ Groq LLM    │  llama-3.3-70b-versatile (JSON Mode)
                             │ Generation  │
                             └──────┬──────┘
                                    ▼
                             ┌─────────────┐
                             │ Grounding   │── Unbacked claim? Retry or Abstain
                             │ Validator   │
                             └──────┬──────┘
                                    ▼
                             Structured JSON + Source Citations
```

---

## 2. Key Engineering Improvements over Baseline

| Feature | Baseline (`rag_core.py`) | Production Upgrade | Rationale |
|---|---|---|---|
| **Ingestion** | In-memory on startup | Offline pre-built disk indexes | Drops startup from minutes to milliseconds |
| **Hybrid Fusion** | `list(set(dense + bm25))` | **Reciprocal Rank Fusion (RRF)** | RRF preserves ranking order and boosts intersecting top hits without score scale distortion |
| **BM25 Tokenizer** | `text.split()` | Normalized + lowercase + punctuation handling | Fixes missing keyword matches on formatted text |
| **FAISS Index** | `IndexFlatL2` | `IndexFlatIP` (Cosine Sim) | Normalized embeddings with inner product give intuitive [0, 1] cosine scores |
| **Query Rewriting** | Always calls LLM (~400 ms) | **Conditional Rewriting** | Bypasses LLM for self-contained queries, saving ~400ms per query |
| **LLM Output** | `response.split("Summary:")` | **JSON Mode + Regex Fallback** | Eliminates crashes from malformed LLM outputs |
| **Guardrails** | None | Input safety, domain relevance, confidence gate, grounding check | System knows when NOT to answer |
| **Orchestration** | Monolithic class method | **LangGraph StateGraph** | Explicit state machine with conditional routing |
| **Voice / STT** | None | **ElevenLabs STT** | Full voice-to-text pipeline with isolated STT vs RAG latency reporting |

---

## 3. Quickstart

### Setup & Installation
```bash
# 1. Clone / Navigate
cd goa_task_2

# 2. Configure Environment
cp .env.example .env
# Edit .env and enter your GROQ_API_KEY and ELEVENLABS_API_KEY

# 3. Install Dependencies
pip install -r requirements.txt
```

### Build Offline Index
```bash
# Ingest 10,000 samples from MSMARCO-XI with passthrough chunking:
python -m scripts.build_index --sample 10000 --strategy passthrough
```

### Run API Server
```bash
# Start FastAPI service
python -m app.main
```

---

## 4. API Documentation

### Text Query Endpoint
`POST /query`
```json
{
  "query": "What is a corporation?",
  "conversation_history": [],
  "query_mode": "normal"
}
```

**Response:**
```json
{
  "answer": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
  "summary": "Corporation is a legal entity distinct from its members.",
  "grounded": true,
  "sources": [
    {
      "chunk_id": "1102432_5_chunk_0",
      "passage_id": "1102432_5",
      "score": 6.8412
    }
  ],
  "latency_ms": 115.42
}
```

### Voice Query Endpoint
`POST /voice`
- Form-data payload with `file: <audio_file.wav>`

**Response:**
```json
{
  "transcript": "what is a corporation",
  "answer": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
  "summary": "Corporation is a legally recognized single entity.",
  "grounded": true,
  "sources": [...],
  "stt_latency_ms": 320.15,
  "rag_latency_ms": 115.42,
  "total_voice_latency_ms": 435.57
}
```

---

## 5. Benchmarking & Evaluation

### Run Automated Unit Tests
```bash
pytest tests/ -v
```

### Run Retrieval Quality & Latency Benchmark
```bash
# Evaluate Recall@K, MRR, and P50/P70/P100 latency percentiles
python -m scripts.run_benchmark --type all --num-queries 50
```
