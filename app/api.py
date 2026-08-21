"""
FastAPI Route Definitions with Real-Time Token Streaming

Endpoints:
    GET  /              -> Interactive Web Test UI (Real-Time Token Streaming + Voice Recording)
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

@router.get("/", response_class=HTMLResponse)
def root():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Goa Hackathon 2026 — Real-Time Streaming Voice RAG</title>
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
        .tab-bar {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--subtext);
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            border-radius: 6px;
        }
        .tab-btn.active {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
        }
        input, select, button {
            width: 100%;
            padding: 0.75rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #0f172a;
            color: var(--text);
            box-sizing: border-box;
            font-size: 1rem;
        }
        button.action-btn {
            background: var(--accent);
            color: #0f172a;
            font-weight: 700;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        button.action-btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .voice-controls {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        .record-btn {
            flex: 1;
            background: #e11d48;
            color: white;
            font-weight: 700;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .record-btn.recording {
            background: #9f1239;
            animation: pulse 1.2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: bold;
            background: rgba(56, 189, 248, 0.2);
            color: var(--accent);
            margin-left: 0.4rem;
        }
        .badge-green {
            background: rgba(52, 211, 153, 0.2);
            color: var(--accent-green);
        }
        .badge-purple {
            background: rgba(192, 132, 252, 0.2);
            color: var(--accent-purple);
        }
        .result-box {
            display: none;
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .stream-cursor {
            display: inline-block;
            width: 8px;
            height: 1.1rem;
            background: var(--accent);
            vertical-align: middle;
            animation: blink 0.8s infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        .source-tag {
            background: #1e293b;
            border: 1px solid #334155;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            display: inline-block;
            margin-top: 0.5rem;
        }
        .links a {
            color: var(--accent);
            text-decoration: none;
            margin-right: 1.5rem;
            font-size: 0.9rem;
        }
        .links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Real-Time Streaming Voice RAG</h1>
        <p class="subtitle">Goa Hackathon 2026 — Task 2 | Sarvam AI STT + FAISS + BM25 + Real-Time Token Streaming</p>
        
        <div class="links" style="margin-bottom: 1rem;">
            <a href="/docs" target="_blank">📖 OpenAPI Docs (/docs)</a>
            <a href="/health" target="_blank">🩺 Health (/health)</a>
        </div>

        <div class="card">
            <div class="tab-bar">
                <button class="tab-btn active" id="tabVoice" onclick="switchTab('voice')">🎙️ Voice Input (Streaming)</button>
                <button class="tab-btn" id="tabText" onclick="switchTab('text')">📝 Text Query (Streaming)</button>
            </div>

            <!-- Voice Panel -->
            <div id="panelVoice">
                <p style="color: var(--subtext); margin-top: 0;">Speak into your microphone or upload audio. Transcribes via <strong>Sarvam AI</strong> and streams answer tokens in real-time:</p>
                
                <div class="voice-controls">
                    <button id="btnRecord" class="record-btn" onclick="toggleRecording()">
                        <span id="recordIcon">🔴</span> <span id="recordText">Start Recording</span>
                    </button>
                </div>
                <div id="recordingTimer" style="display: none; color: var(--accent-red); margin-bottom: 1rem; font-weight: 600;">
                    🎙️ Recording... Speak into your microphone, then click Stop.
                </div>

                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed var(--border);">
                    <label for="audioFile">Or upload audio file (.wav, .mp3):</label>
                    <input type="file" id="audioFile" accept="audio/*">
                    <button class="action-btn" onclick="uploadAudioFile()">📤 Stream Uploaded Audio</button>
                </div>
            </div>

            <!-- Text Panel -->
            <div id="panelText" style="display: none;">
                <label for="queryInput">Query:</label>
                <input type="text" id="queryInput" placeholder="e.g. What is a corporation?" value="What is a corporation?">
                
                <label for="modeSelect">Retrieval Mode:</label>
                <select id="modeSelect">
                    <option value="normal">Normal (Fast RRF)</option>
                    <option value="query_rewrite">Force Rewrite</option>
                    <option value="hyde">HyDE</option>
                </select>
                
                <button class="action-btn" onclick="submitStreamingTextQuery()">⚡ Stream Answer Tokens</button>
            </div>

            <!-- Live Stream Result Box -->
            <div id="resultBox" class="result-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>Output:</strong>
                    <div id="latencyBadges"></div>
                </div>

                <div id="transcriptSection" style="display: none; margin-bottom: 0.75rem; padding: 0.5rem 0.75rem; background: #1e293b; border-radius: 6px;">
                    <strong style="color: var(--accent);">🎙️ Sarvam Transcript:</strong>
                    <span id="transcriptText" style="margin-left: 0.5rem; color: #fff;"></span>
                </div>

                <div>
                    <strong>Answer:</strong>
                    <p id="answerContainer" style="margin-top: 0.5rem; line-height: 1.5; font-size: 1.05rem;">
                        <span id="answerText"></span><span id="cursor" class="stream-cursor" style="display: none;"></span>
                    </p>
                </div>

                <div id="sourcesContainer" style="margin-top: 0.75rem;"></div>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        function switchTab(tab) {
            document.getElementById('tabVoice').classList.toggle('active', tab === 'voice');
            document.getElementById('tabText').classList.toggle('active', tab === 'text');
            document.getElementById('panelVoice').style.display = tab === 'voice' ? 'block' : 'none';
            document.getElementById('panelText').style.display = tab === 'text' ? 'block' : 'none';
        }

        async function toggleRecording() {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await streamAudio(audioBlob);
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById('recordIcon').innerText = '⏹️';
                    document.getElementById('recordText').innerText = 'Stop & Stream Answer';
                    document.getElementById('btnRecord').classList.add('recording');
                    document.getElementById('recordingTimer').style.display = 'block';
                } catch (err) {
                    alert('Microphone access denied: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('recordIcon').innerText = '🔴';
                document.getElementById('recordText').innerText = 'Start Recording';
                document.getElementById('btnRecord').classList.remove('recording');
                document.getElementById('recordingTimer').style.display = 'none';
            }
        }

        async function uploadAudioFile() {
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select an audio file first.');
                return;
            }
            await streamAudio(fileInput.files[0]);
        }

        async function streamAudio(blob) {
            const resBox = document.getElementById('resultBox');
            resBox.style.display = 'block';
            document.getElementById('transcriptSection').style.display = 'none';
            document.getElementById('answerText').innerText = '';
            document.getElementById('cursor').style.display = 'inline-block';
            document.getElementById('sourcesContainer').innerHTML = '';
            document.getElementById('latencyBadges').innerHTML = '<span class="badge">🎙️ Transcribing with Sarvam...</span>';

            const formData = new FormData();
            formData.append('file', blob, 'audio.wav');
            formData.append('query_mode', 'normal');

            try {
                const response = await fetch('/voice/stream', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errJson = await response.json().catch(() => ({}));
                    throw new Error(errJson.detail || 'Server returned status ' + response.status);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'stt') {
                            document.getElementById('transcriptSection').style.display = 'block';
                            document.getElementById('transcriptText').innerText = data.transcript;
                            document.getElementById('latencyBadges').innerHTML = 
                                `<span class="badge badge-green">STT: ${data.stt_latency_ms} ms</span> ` +
                                `<span class="badge badge-purple">Retrieving...</span>`;
                        } else if (data.type === 'metadata') {
                            renderSources(data.sources);
                        } else if (data.type === 'token') {
                            document.getElementById('answerText').innerText += data.token;
                            if (data.ttft_ms) {
                                document.getElementById('latencyBadges').innerHTML += 
                                    `<span class="badge badge-purple">TTFT: ${data.ttft_ms} ms ⚡</span>`;
                            }
                        } else if (data.type === 'done') {
                            document.getElementById('cursor').style.display = 'none';
                            document.getElementById('latencyBadges').innerHTML += 
                                `<span class="badge">Total: ${data.total_voice_latency_ms} ms</span>`;
                        }
                    }
                }
            } catch (err) {
                document.getElementById('answerText').innerText = 'Error: ' + err.message;
                document.getElementById('cursor').style.display = 'none';
            }
        }

        async function submitStreamingTextQuery() {
            const query = document.getElementById('queryInput').value.trim();
            const mode = document.getElementById('modeSelect').value;
            if (!query) return;

            const resBox = document.getElementById('resultBox');
            resBox.style.display = 'block';
            document.getElementById('transcriptSection').style.display = 'none';
            document.getElementById('answerText').innerText = '';
            document.getElementById('cursor').style.display = 'inline-block';
            document.getElementById('sourcesContainer').innerHTML = '';
            document.getElementById('latencyBadges').innerHTML = '<span class="badge">Retrieving...</span>';

            try {
                const response = await fetch('/query/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query, query_mode: mode, conversation_history: []})
                });

                if (!response.ok) {
                    const errJson = await response.json().catch(() => ({}));
                    throw new Error(errJson.detail || 'Server returned status ' + response.status);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'metadata') {
                            document.getElementById('latencyBadges').innerHTML = 
                                `<span class="badge badge-green">Retrieval: ${data.retrieval_latency_ms} ms</span>`;
                            renderSources(data.sources);
                        } else if (data.type === 'token') {
                            document.getElementById('answerText').innerText += data.token;
                            if (data.ttft_ms) {
                                document.getElementById('latencyBadges').innerHTML += 
                                    `<span class="badge badge-purple">TTFT: ${data.ttft_ms} ms ⚡</span>`;
                            }
                        } else if (data.type === 'done') {
                            document.getElementById('cursor').style.display = 'none';
                            document.getElementById('latencyBadges').innerHTML += 
                                `<span class="badge">Total: ${data.total_latency_ms} ms</span>`;
                        }
                    }
                }
            } catch (err) {
                document.getElementById('answerText').innerText = 'Error: ' + err.message;
                document.getElementById('cursor').style.display = 'none';
            }
        }

        function renderSources(sources) {
            const sourcesDiv = document.getElementById('sourcesContainer');
            if (sources && sources.length > 0) {
                sourcesDiv.innerHTML = '<strong>Sources:</strong><br/>' + 
                    sources.map(s => `<span class="source-tag">📄 ${s.chunk_id}</span>`).join('');
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": _pipeline is not None,
        "indexed_chunks": len(_pipeline.chunks) if _pipeline else 0,
    }


@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not yet initialized.")

    try:
        res = _pipeline.run(
            query=req.query,
            conversation_history=req.conversation_history,
            query_mode=req.query_mode,
        )

        return QueryResponse(
            answer=res["answer"],
            summary=res["summary"],
            grounded=res["grounded"],
            sources=[SourceCitation(**s) for s in res.get("sources", [])],
            latency_ms=res["total_latency_ms"],
        )
    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
def query_rag_stream(req: QueryRequest):
    """Server-Sent Events (SSE) real-time streaming endpoint for text queries."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not yet initialized.")

    def event_generator():
        for event in _pipeline.run_stream(
            query=req.query,
            conversation_history=req.conversation_history,
            query_mode=req.query_mode,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/voice", response_model=VoiceResponse)
async def voice_rag(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    query_mode: str = Form("normal"),
):
    if _pipeline is None or _stt is None:
        raise HTTPException(status_code=503, detail="Services not yet initialized.")

    t0_voice = time.perf_counter()
    
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        # 1. STT (Sarvam AI saaras:v3)
        transcript, stt_latency_ms = _stt.transcribe(audio_bytes)

        # 2. RAG Pipeline
        rag_res = _pipeline.run(query=transcript, query_mode=query_mode)
        
        total_voice_ms = (time.perf_counter() - t0_voice) * 1000

        return VoiceResponse(
            transcript=transcript,
            answer=rag_res["answer"],
            summary=rag_res["summary"],
            grounded=rag_res["grounded"],
            sources=[SourceCitation(**s) for s in rag_res.get("sources", [])],
            stt_latency_ms=round(stt_latency_ms, 2),
            rag_latency_ms=rag_res["total_latency_ms"],
            total_voice_latency_ms=round(total_voice_ms, 2),
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error("Error in voice RAG pipeline: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/stream")
async def voice_rag_stream(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    query_mode: str = Form("normal"),
):
    """Server-Sent Events (SSE) real-time streaming endpoint for voice queries."""
    if _pipeline is None or _stt is None:
        raise HTTPException(status_code=503, detail="Services not yet initialized.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    t0_voice = time.perf_counter()

    def event_generator():
        try:
            transcript, stt_latency_ms = _stt.transcribe(audio_bytes)
        except Exception as e:
            logger.error("Voice transcription failed: %s", e)
            yield f"data: {json.dumps({'type': 'token', 'token': f'Transcription Error: {e}', 'ttft_ms': 0.1})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'total_voice_latency_ms': 0.0})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'stt', 'transcript': transcript, 'stt_latency_ms': round(stt_latency_ms, 2)})}\n\n"
        
        for event in _pipeline.run_stream(query=transcript, query_mode=query_mode):
            if event.get("type") == "done":
                total_voice_ms = (time.perf_counter() - t0_voice) * 1000
                event["total_voice_latency_ms"] = round(total_voice_ms, 2)
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
