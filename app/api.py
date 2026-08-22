"""
FastAPI Route Definitions with Real-Time Token Streaming

Endpoints:
    GET  /              -> Interactive Web Test UI (Real-Time Token Streaming + Voice Recording)
    GET  /favicon.ico   -> Browser icon handler
    GET  /health        -> Service health status
    POST /query         -> Synchronous RAG query
    POST /query/stream  -> Streaming RAG query (Server-Sent Events)
    POST /voice         -> Synchronous voice query (Sarvam STT -> RAG)
    POST /voice/stream  -> Streaming voice query (Sarvam STT -> Real-Time Streaming RAG)
"""

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from pipeline.graph import RAGPipelineGraph
from voice.stt import SarvamSTT

logger = logging.getLogger(__name__)

router = APIRouter()

# Global pipeline instance
_pipeline: Optional[RAGPipelineGraph] = None
_stt: Optional[SarvamSTT] = None


def set_pipeline_instance(pipeline: RAGPipelineGraph, stt: SarvamSTT):
    global _pipeline, _stt
    _pipeline = pipeline
    _stt = stt


# ── Schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question text", example="What is a corporation?")
    conversation_history: list[dict] = Field(default_factory=list, description="Previous conversation turns")
    query_mode: str = Field("normal", description="Retrieval mode: normal, query_rewrite, or hyde")


class SourceCitation(BaseModel):
    chunk_id: str
    passage_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    summary: str
    grounded: bool
    sources: list[SourceCitation]
    latency_ms: float


class VoiceResponse(BaseModel):
    transcript: str
    answer: str
    summary: str
    grounded: bool
    sources: list[SourceCitation]
    stt_latency_ms: float
    rag_latency_ms: float
    total_voice_latency_ms: float


# ── HTML UI ───────────────────────────────────────────────────────────

@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(content="", status_code=204)


@router.get("/", response_class=HTMLResponse)
def root():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Goa Hackathon 2026 — Real-Time Streaming Voice RAG</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎙️</text></svg>">
    <style>
        :root {
            --bg: #0b1120;
            --card: #1e293b;
            --accent: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
            --text: #f8fafc;
            --subtext: #94a3b8;
            --border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
        }
        .container {
            max-width: 820px;
            width: 100%;
        }
        h1 { color: var(--accent); margin-bottom: 0.25rem; font-size: 1.8rem; }
        p.subtitle { color: var(--subtext); margin-top: 0; margin-bottom: 1.5rem; font-size: 0.95rem; }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        .tab-nav {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }
        .tab-btn {
            background: none;
            border: none;
            color: var(--subtext);
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        .tab-btn.active {
            color: var(--accent);
            background: rgba(56, 189, 248, 0.1);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        textarea, input[type="text"] {
            width: 100%;
            box-sizing: border-box;
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 0.75rem;
            font-size: 1rem;
            margin-bottom: 1rem;
            resize: vertical;
        }
        button.action-btn {
            background: var(--accent);
            color: #0b1120;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        button.action-btn:hover { opacity: 0.9; }
        button.record-btn {
            background: var(--accent-red);
            color: white;
        }
        button.record-btn.recording {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.04); }
            100% { transform: scale(1); }
        }
        .response-box {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1rem;
            min-height: 80px;
            white-space: pre-wrap;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .metrics-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .metric-badge {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.4rem 0.75rem;
            font-size: 0.85rem;
        }
        .metric-badge span {
            font-weight: bold;
            color: var(--accent-green);
        }
        .streaming-cursor {
            display: inline-block;
            width: 8px;
            height: 16px;
            background: var(--accent);
            margin-left: 4px;
            vertical-align: middle;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Goa Hackathon 2026 — Real-Time Streaming Voice RAG</h1>
        <p class="subtitle">Sub-200ms RAG | Sarvam AI STT (saaras:v3) | SQLite FTS5 (23M+ Passages) | Groq (allam-2-7b)</p>

        <div class="card">
            <div class="tab-nav">
                <button class="tab-btn active" onclick="switchTab('voice-tab')">🎙️ Voice Input (Live Recording)</button>
                <button class="tab-btn" onclick="switchTab('text-tab')">📝 Text Query (Streaming)</button>
            </div>

            <!-- Voice Tab -->
            <div id="voice-tab" class="tab-content active">
                <p style="color: var(--subtext); font-size: 0.9rem; margin-top:0;">Click to record your voice. Your speech will be transcribed via Sarvam AI and streamed in real-time.</p>
                <button id="record-btn" class="action-btn record-btn" onclick="toggleRecording()">
                    <span id="record-icon">🎤</span> <span id="record-text">Start Recording</span>
                </button>
                <span id="recording-timer" style="margin-left: 1rem; color: var(--accent-red); font-weight: bold; display: none;">🔴 Recording...</span>
            </div>

            <!-- Text Tab -->
            <div id="text-tab" class="tab-content">
                <textarea id="text-query" rows="2" placeholder="Ask any question (e.g. 'What is a corporation?', 'symptoms of borderline personality disorder')...">What is a corporation?</textarea>
                <button class="action-btn" onclick="sendTextQuery()">⚡ Stream Answer</button>
            </div>
        </div>

        <div class="card">
            <h3 style="margin-top:0; color: var(--text); font-size: 1.1rem;">Response Stream:</h3>
            <div id="transcript-container" style="display: none; margin-bottom: 0.75rem; color: var(--accent-purple); font-size: 0.95rem;">
                <strong>Transcribed Speech:</strong> <span id="transcript-text"></span>
            </div>
            <div id="response-box" class="response-box">
                <span id="response-text" style="color: var(--subtext);">Waiting for input...</span>
                <span id="cursor" class="streaming-cursor" style="display:none;"></span>
            </div>

            <div id="metrics-container" class="metrics-grid" style="display: none;">
                <div class="metric-badge">STT Latency: <span id="metric-stt">-</span></div>
                <div class="metric-badge">Retrieval: <span id="metric-retrieval">-</span></div>
                <div class="metric-badge">Time to First Token (TTFT): <span id="metric-ttft">-</span></div>
                <div class="metric-badge">Total Pipeline Latency: <span id="metric-total">-</span></div>
            </div>

            <div id="sources-container" style="display: none; margin-top: 1rem; font-size: 0.85rem; color: var(--subtext);">
                <strong>Retrieved Grounding Sources:</strong>
                <ul id="sources-list" style="margin: 0.5rem 0 0 1.25rem; padding: 0;"></ul>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        async function toggleRecording() {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                    audioChunks = [];

                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        await streamVoice(audioBlob);
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById('record-btn').classList.add('recording');
                    document.getElementById('record-text').innerText = 'Stop & Stream Answer';
                    document.getElementById('recording-timer').style.display = 'inline';
                } catch (err) {
                    alert('Microphone access denied or not available: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('record-btn').classList.remove('recording');
                document.getElementById('record-text').innerText = 'Start Recording';
                document.getElementById('recording-timer').style.display = 'none';
            }
        }

        async function streamVoice(audioBlob) {
            resetUI();
            document.getElementById('response-text').innerText = '';
            document.getElementById('cursor').style.display = 'inline-block';

            const formData = new FormData();
            formData.append('audio', audioBlob, 'speech.webm');
            formData.append('language_code', 'en-IN');

            try {
                const response = await fetch('/voice/stream', {
                    method: 'POST',
                    body: formData
                });

                await readSSE(response);
            } catch (err) {
                document.getElementById('response-text').innerText = 'Error: ' + err.message;
                document.getElementById('cursor').style.display = 'none';
            }
        }

        async function sendTextQuery() {
            const query = document.getElementById('text-query').value.trim();
            if (!query) return;

            resetUI();
            document.getElementById('response-text').innerText = '';
            document.getElementById('cursor').style.display = 'inline-block';

            try {
                const response = await fetch('/query/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, query_mode: 'normal' })
                });

                await readSSE(response);
            } catch (err) {
                document.getElementById('response-text').innerText = 'Error: ' + err.message;
                document.getElementById('cursor').style.display = 'none';
            }
        }

        async function readSSE(response) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const nl = String.fromCharCode(10);
                const lines = buffer.split(nl);
                buffer = lines.pop(); // keep last incomplete chunk

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('data:')) {
                        const jsonStr = trimmed.substring(5).trim();
                        if (!jsonStr) continue;
                        try {
                            const data = JSON.parse(jsonStr);
                            handleSSEEvent(data);
                        } catch (e) {
                            console.error('SSE JSON parse error:', e, jsonStr);
                        }
                    }
                }
            }
            document.getElementById('cursor').style.display = 'none';
        }

        function handleSSEEvent(data) {
            if (data.type === 'metadata') {
                if (data.transcript) {
                    document.getElementById('transcript-container').style.display = 'block';
                    document.getElementById('transcript-text').innerText = `"${data.transcript}"`;
                }
                document.getElementById('metrics-container').style.display = 'flex';
                if (data.stt_latency_ms) {
                    document.getElementById('metric-stt').innerText = `${data.stt_latency_ms.toFixed(1)} ms`;
                }
                document.getElementById('metric-retrieval').innerText = `${data.retrieval_latency_ms.toFixed(1)} ms`;

                if (data.sources && data.sources.length > 0) {
                    const list = document.getElementById('sources-list');
                    list.innerHTML = '';
                    data.sources.forEach(s => {
                        const li = document.createElement('li');
                        li.innerText = `[${s.passage_id || s.chunk_id}] (Score: ${s.score})`;
                        list.appendChild(li);
                    });
                    document.getElementById('sources-container').style.display = 'block';
                }
            } else if (data.type === 'token') {
                document.getElementById('response-text').innerText += data.token;
                if (data.ttft_ms) {
                    document.getElementById('metric-ttft').innerText = `${data.ttft_ms.toFixed(1)} ms ⚡`;
                }
            } else if (data.type === 'done') {
                document.getElementById('cursor').style.display = 'none';
                document.getElementById('metric-total').innerText = `${data.total_latency_ms.toFixed(1)} ms`;
            }
        }

        function resetUI() {
            document.getElementById('transcript-container').style.display = 'none';
            document.getElementById('sources-container').style.display = 'none';
            document.getElementById('metrics-container').style.display = 'none';
            document.getElementById('metric-stt').innerText = '-';
            document.getElementById('metric-ttft').innerText = '-';
            document.getElementById('metric-total').innerText = '-';
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# ── API Endpoints ─────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Health check returning database connection and retriever status."""
    if _pipeline is None:
        return {"status": "starting", "pipeline_ready": False}

    fts_count = _pipeline.fts_retriever.count() if _pipeline.fts_retriever else 0
    return {
        "status": "healthy",
        "pipeline_ready": True,
        "sqlite_fts_passages": fts_count,
        "in_memory_chunks": len(_pipeline.chunks),
        "dense_retriever": _pipeline.dense_retriever is not None,
    }


@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """Synchronous JSON RAG Query endpoint."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    t0 = time.perf_counter()
    result = _pipeline.run(
        query=req.query,
        conversation_history=req.conversation_history,
        query_mode=req.query_mode,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        answer=result.get("answer", ""),
        summary=result.get("summary", ""),
        grounded=result.get("grounded", False),
        sources=[
            SourceCitation(
                chunk_id=s.get("chunk_id", ""),
                passage_id=s.get("passage_id", ""),
                score=s.get("score", 0.0),
            )
            for s in result.get("sources", [])
        ],
        latency_ms=round(elapsed, 2),
    )


@router.post("/query/stream")
def query_rag_stream(req: QueryRequest):
    """Server-Sent Events (SSE) Streaming RAG Query endpoint."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    def event_stream():
        for event in _pipeline.run_stream(
            query=req.query,
            conversation_history=req.conversation_history,
            query_mode=req.query_mode,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/voice", response_model=VoiceResponse)
async def query_voice(
    audio: UploadFile = File(..., description="Audio recording file (WAV/WEBM/MP3)"),
    language_code: str = Form("en-IN"),
):
    """Synchronous Voice Query: Sarvam STT -> RAG."""
    if _pipeline is None or _stt is None:
        raise HTTPException(status_code=503, detail="Voice pipeline not initialized")

    t0 = time.perf_counter()
    audio_bytes = await audio.read()

    try:
        transcript, stt_latency = _stt.transcribe(audio_bytes, language_code=language_code)
    except Exception as e:
        logger.error("Voice transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Voice transcription failed: {e}")

    rag_res = _pipeline.run(query=transcript)
    total_latency = (time.perf_counter() - t0) * 1000

    return VoiceResponse(
        transcript=transcript,
        answer=rag_res.get("answer", ""),
        summary=rag_res.get("summary", ""),
        grounded=rag_res.get("grounded", False),
        sources=[
            SourceCitation(
                chunk_id=s.get("chunk_id", ""),
                passage_id=s.get("passage_id", ""),
                score=s.get("score", 0.0),
            )
            for s in rag_res.get("sources", [])
        ],
        stt_latency_ms=round(stt_latency, 2),
        rag_latency_ms=rag_res.get("total_latency_ms", 0.0),
        total_voice_latency_ms=round(total_latency, 2),
    )


@router.post("/voice/stream")
async def query_voice_stream(
    audio: UploadFile = File(..., description="Audio recording file (WAV/WEBM/MP3)"),
    language_code: str = Form("en-IN"),
):
    """Streaming Voice Query: Sarvam STT -> Streaming RAG (Server-Sent Events)."""
    if _pipeline is None or _stt is None:
        raise HTTPException(status_code=503, detail="Voice pipeline not initialized")

    t0 = time.perf_counter()
    audio_bytes = await audio.read()

    try:
        transcript, stt_latency = _stt.transcribe(audio_bytes, language_code=language_code)
    except Exception as e:
        logger.error("Voice transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Voice transcription failed: {e}")

    def event_stream():
        # First yield the STT transcript and metadata
        first_event = True
        for event in _pipeline.run_stream(query=transcript):
            if first_event and event.get("type") == "metadata":
                event["transcript"] = transcript
                event["stt_latency_ms"] = round(stt_latency, 2)
                first_event = False
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
