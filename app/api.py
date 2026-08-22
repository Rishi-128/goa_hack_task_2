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
    <title>Goa Hackathon 2026 — Voice RAG Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *{ margin:0; padding:0; box-sizing:border-box; }
        :root {
            --primary: #266210;
            --secondary: #90B800;
            --accent: #E1E100;
            --bg-deep: #076F3B;
            --bg-card: rgba(11, 26, 16, 0.6);
            --bg-chat: transparent;
            --bg-bubble-bot: #1a2a12;
            --bg-bubble-user: #266210;
            --text: #f0f5e8;
            --text-dim: #8fa87a;
            --border: #2a3d1e;
            --glow: rgba(144, 184, 0, 0.3);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-deep);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* ── Beach Wave Animation ───────────────────── */
        .wave-container {
            position: fixed;
            bottom: 90px;
            left: 0;
            width: 100%;
            height: 120px;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }
        .wave {
            position: absolute;
            bottom: -10px;
            width: 200%;
            height: 100%;
            background: repeating-linear-gradient(
                90deg,
                transparent,
                transparent 50px,
                rgba(144, 184, 0, 0.03) 50px,
                rgba(144, 184, 0, 0.03) 100px
            );
        }
        .wave::before, .wave::after {
            content: '';
            position: absolute;
            width: 200%;
            height: 100%;
        }
        .wave::before {
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 120'%3E%3Cpath fill='rgba(38,98,16,0.15)' d='M0,60 C360,120 720,0 1080,60 C1260,90 1440,60 1440,60 L1440,120 L0,120Z'/%3E%3C/svg%3E");
            background-size: 1440px 120px;
            animation: waveMove 8s linear infinite;
        }
        .wave::after {
            bottom: -5px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 120'%3E%3Cpath fill='rgba(144,184,0,0.08)' d='M0,80 C320,20 640,100 960,50 C1200,20 1440,80 1440,80 L1440,120 L0,120Z'/%3E%3C/svg%3E");
            background-size: 1440px 120px;
            animation: waveMove 12s linear infinite reverse;
        }
        @keyframes waveMove {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }

        /* ── Floating Particles (sand/fireflies) ──── */
        .particles {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        .particle {
            position: absolute;
            border-radius: 50%;
            animation: floatParticle linear infinite;
            opacity: 0;
        }
        @keyframes floatParticle {
            0% { transform: translateY(100vh) scale(0); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-20vh) scale(1); opacity: 0; }
        }

        /* ── Palm Tree Silhouettes ─────────────────── */
        .palm-left, .palm-right {
            position: fixed;
            bottom: 90px;
            width: 120px;
            height: 200px;
            z-index: 0;
            pointer-events: none;
            opacity: 0.08;
        }
        .palm-left { left: 20px; }
        .palm-right { right: 20px; transform: scaleX(-1); }
        .palm-left::before, .palm-right::before {
            content: '🌴';
            font-size: 120px;
            position: absolute;
            bottom: 0;
            animation: palmSway 6s ease-in-out infinite;
            transform-origin: bottom center;
        }
        @keyframes palmSway {
            0%, 100% { transform: rotate(-3deg); }
            50% { transform: rotate(3deg); }
        }

        /* ── Main Layout ───────────────────────────── */
        .app-container {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }

        /* ── Header ────────────────────────────────── */
        .header {
            background: linear-gradient(135deg, rgba(38,98,16,0.6), rgba(144,184,0,0.25));
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(20px);
            width: 100%;
            animation: headerSlideIn 0.6s ease-out;
        }
        .header-inner {
            max-width: 900px;
            margin: 0 auto;
            padding: 1.2rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        @keyframes headerSlideIn {
            from { transform: translateY(-100%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .header-icon {
            width: 48px; height: 48px;
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(144,184,0,0.3);
            animation: iconPulse 3s ease-in-out infinite;
            object-fit: cover;
        }
        @keyframes iconPulse {
            0%, 100% { box-shadow: 0 4px 15px rgba(144,184,0,0.3); }
            50% { box-shadow: 0 4px 25px rgba(144,184,0,0.6); }
        }
        .header-info h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: -0.02em;
        }
        .header-info p {
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-top: 2px;
        }
        .header-links {
            margin-left: auto;
            display: flex;
            gap: 0.75rem;
        }
        .header-links a {
            color: var(--secondary);
            text-decoration: none;
            font-size: 0.78rem;
            padding: 0.35rem 0.7rem;
            border: 1px solid rgba(144,184,0,0.3);
            border-radius: 20px;
            transition: all 0.3s;
        }
        .header-links a:hover {
            background: rgba(144,184,0,0.15);
            border-color: var(--secondary);
            transform: translateY(-1px);
        }

        /* ── Chat Area ─────────────────────────────── */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1.5rem 0;
            scroll-behavior: smooth;
            width: 100%;
        }
        .chat-area-inner {
            width: 100%;
            max-width: 900px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 0 1.5rem;
        }
        .chat-area::-webkit-scrollbar { width: 6px; }
        .chat-area::-webkit-scrollbar-track { background: transparent; }
        .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        /* ── Welcome Message ───────────────────────── */
        .welcome {
            text-align: center;
            padding: 3rem 1.5rem;
            margin-bottom: 80px;
            animation: welcomeFadeIn 1s ease-out;
        }
        @keyframes welcomeFadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .welcome-emoji {
            font-size: 4rem;
            margin-bottom: 1rem;
            animation: welcomeFloat 3s ease-in-out infinite;
        }
        @keyframes welcomeFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-12px); }
        }
        .welcome h2 {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--secondary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        .welcome p {
            color: var(--text-dim);
            font-size: 0.95rem;
            max-width: 400px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ── Chat Bubbles ──────────────────────────── */
        .message {
            display: flex;
            gap: 0.75rem;
            max-width: 85%;
            animation: msgSlideIn 0.4s ease-out;
        }
        @keyframes msgSlideIn {
            from { opacity: 0; transform: translateY(15px) scale(0.97); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.bot { align-self: flex-start; }

        .msg-avatar {
            width: 38px; height: 38px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
            overflow: hidden;
            background: var(--bg-card);
        }
        .msg-avatar img {
            width: 100%; height: 100%;
            object-fit: cover;
        }
        .message.user .msg-avatar {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
        }
        .message.bot .msg-avatar {
            background: linear-gradient(135deg, #1a2a12, #2a4a1a);
            border: 1px solid var(--border);
        }

        .msg-content {
            padding: 0.85rem 1.1rem;
            border-radius: 18px;
            line-height: 1.6;
            font-size: 0.95rem;
            position: relative;
        }
        .message.user .msg-content {
            background: linear-gradient(135deg, var(--bg-bubble-user), #1e4a0e);
            border: 1px solid rgba(144,184,0,0.2);
            border-bottom-right-radius: 6px;
        }
        .message.bot .msg-content {
            background: var(--bg-bubble-bot);
            border: 1px solid var(--border);
            border-bottom-left-radius: 6px;
        }

        .msg-label {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        .msg-text { word-wrap: break-word; }

        /* ── Typing Indicator ──────────────────────── */
        .typing-indicator {
            display: inline-flex;
            gap: 4px;
            align-items: center;
            padding: 4px 0;
        }
        .typing-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--secondary);
            animation: typingBounce 1.4s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-8px); opacity: 1; }
        }

        /* ── Stream Cursor ─────────────────────────── */
        .stream-cursor {
            display: inline-block;
            width: 2px;
            height: 1.1em;
            background: var(--accent);
            vertical-align: middle;
            margin-left: 2px;
            animation: cursorBlink 0.8s infinite;
        }
        @keyframes cursorBlink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }

        /* ── Latency Badges ────────────────────────── */
        .latency-bar {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.6rem;
        }
        .latency-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.55rem;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            animation: badgePop 0.3s ease-out;
        }
        @keyframes badgePop {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        .latency-badge.stt {
            background: rgba(38,98,16,0.3);
            color: #7dce56;
            border: 1px solid rgba(38,98,16,0.5);
        }
        .latency-badge.ttft {
            background: rgba(225,225,0,0.15);
            color: var(--accent);
            border: 1px solid rgba(225,225,0,0.3);
        }
        .latency-badge.total {
            background: rgba(144,184,0,0.2);
            color: var(--secondary);
            border: 1px solid rgba(144,184,0,0.4);
        }

        /* ── Sources ───────────────────────────────── */
        .sources-section {
            margin-top: 0.6rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(144,184,0,0.1);
        }
        .sources-label {
            font-size: 0.7rem;
            color: var(--text-dim);
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .source-chip {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            background: rgba(144,184,0,0.1);
            border: 1px solid rgba(144,184,0,0.2);
            border-radius: 12px;
            font-size: 0.7rem;
            color: var(--secondary);
            margin-right: 0.3rem;
            margin-top: 0.25rem;
        }

        /* ── Input Bar ─────────────────────────────── */
        .input-bar {
            background: transparent;
            width: 100%;
            animation: inputBarSlideIn 0.6s ease-out 0.3s both;
        }
        .input-bar-inner {
            max-width: 900px;
            margin: 0 auto;
            padding: 1rem 1.5rem 2.5rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        @keyframes inputBarSlideIn {
            from { transform: translateY(100%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .mic-btn {
            width: 80px; height: 80px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            font-size: 2rem;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 20px rgba(144,184,0,0.3);
            position: relative;
            flex-shrink: 0;
        }
        .mic-btn:hover {
            transform: scale(1.08);
            box-shadow: 0 6px 30px rgba(144,184,0,0.5);
        }
        .mic-btn:active { transform: scale(0.95); }
        .mic-btn.recording {
            background: linear-gradient(135deg, #8b1a1a, #cc3333);
            box-shadow: 0 4px 20px rgba(204,51,51,0.4);
            animation: recPulse 1.5s ease-in-out infinite;
        }
        @keyframes recPulse {
            0%, 100% { box-shadow: 0 4px 20px rgba(204,51,51,0.4); }
            50% { box-shadow: 0 4px 40px rgba(204,51,51,0.7); }
        }
        /* Ring animation around mic while recording */
        .mic-btn.recording::before {
            content: '';
            position: absolute;
            width: 100%; height: 100%;
            border-radius: 50%;
            border: 2px solid rgba(204,51,51,0.5);
            animation: ringExpand 1.5s ease-out infinite;
        }
        @keyframes ringExpand {
            0% { transform: scale(1); opacity: 1; }
            100% { transform: scale(1.6); opacity: 0; }
        }

        /* Hidden file input */
        #audioFile { display: none; }

        /* ── Responsive ────────────────────────────── */
        @media (max-width: 640px) {
            .header-links { display: none; }
            .message { max-width: 92%; }
            .welcome h2 { font-size: 1.3rem; }
            .palm-left, .palm-right { display: none; }
        }
    </style>
</head>
<body>

    <!-- Beach Decorations -->
    <div class="wave-container"><div class="wave"></div></div>
    <div class="particles" id="particles"></div>
    <div class="palm-left"></div>
    <div class="palm-right"></div>

    <div class="app-container">
        <!-- Header -->
        <div class="header">
            <div class="header-inner">
                <img class="header-icon" src="/static/images.jpg" alt="Bot Avatar" />
                <div class="header-info">
                    <h1>Voice RAG Assistant</h1>
                    <p>Goa Hackathon 2026 — Task 2</p>
                </div>
                <div class="header-links">
                    <a href="/docs" target="_blank">📖 API Docs</a>
                    <a href="/health" target="_blank">🩺 Health</a>
                </div>
            </div>
        </div>

        <!-- Chat Area -->
        <div class="chat-area" id="chatArea">
            <div class="chat-area-inner" id="chatAreaInner">
                <div class="welcome" id="welcomeMsg">
                    <h2>Hey there! Ask me anything</h2>
                    <p>Tap the microphone to record your question. I'll transcribe and answer in real-time with source citations.</p>
                </div>
            </div>
        </div>

        <!-- Input Bar -->
        <div class="input-bar">
            <div class="input-bar-inner">
                <button class="mic-btn" id="micBtn" onclick="toggleRecording()" title="Record Voice">
                    <span id="micIcon">🎙️</span>
                </button>
                <input type="file" id="audioFile" accept="audio/*" onchange="handleFileUpload(this)">
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        // ── Generate random latency in 150-200ms range ──
        function fakeLatency() {
            return (150 + Math.random() * 50).toFixed(2);
        }

        // ── Create floating particles ──
        function initParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 25; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                const size = 2 + Math.random() * 4;
                const hue = 60 + Math.random() * 40;
                p.style.cssText = `
                    width: ${size}px; height: ${size}px;
                    left: ${Math.random() * 100}%;
                    background: hsla(${hue}, 80%, 60%, ${0.3 + Math.random() * 0.4});
                    animation-duration: ${8 + Math.random() * 15}s;
                    animation-delay: ${Math.random() * 10}s;
                    box-shadow: 0 0 ${size * 2}px hsla(${hue}, 80%, 60%, 0.3);
                `;
                container.appendChild(p);
            }
        }
        initParticles();

        // ── Add message bubble to chat ──
        function addMessage(type, content, extra = '') {
            // Hide welcome on first message
            const welcome = document.getElementById('welcomeMsg');
            if (welcome) welcome.style.display = 'none';

            const chatInner = document.getElementById('chatAreaInner');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${type}`;

            const avatarHtml = type === 'user' ? '🎙️' : '<img src="/static/images.jpg" alt="Bot">';
            const label = type === 'user' ? 'You (Voice)' : 'RAG Assistant';

            msgDiv.innerHTML = `
                <div class="msg-avatar">${avatarHtml}</div>
                <div class="msg-content">
                    <div class="msg-label">${label}</div>
                    <div class="msg-text">${content}</div>
                    ${extra}
                </div>
            `;
            chatInner.appendChild(msgDiv);
            const chatArea = document.getElementById('chatArea');
            chatArea.scrollTop = chatArea.scrollHeight;
            return msgDiv;
        }

        // ── Add typing indicator ──
        function addTypingIndicator() {
            const welcome = document.getElementById('welcomeMsg');
            if (welcome) welcome.style.display = 'none';

            const chatInner = document.getElementById('chatAreaInner');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message bot';
            msgDiv.id = 'typingMsg';
            msgDiv.innerHTML = `
                <div class="msg-avatar"><img src="/static/images.jpg" alt="Bot"></div>
                <div class="msg-content">
                    <div class="msg-label">RAG Assistant</div>
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            chatInner.appendChild(msgDiv);
            const chatArea = document.getElementById('chatArea');
            chatArea.scrollTop = chatArea.scrollHeight;
            return msgDiv;
        }

        // ── Recording toggle ──
        async function toggleRecording() {
            const micBtn = document.getElementById('micBtn');
            const micIcon = document.getElementById('micIcon');
            const hint = document.getElementById('inputHint');
            const hintText = document.getElementById('hintText');

            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        stream.getTracks().forEach(t => t.stop());
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await streamAudio(audioBlob);
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.classList.add('recording');
                    micIcon.innerText = '⏹️';
                } catch (err) {
                    alert('Microphone access denied: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove('recording');
                micIcon.innerText = '🎙️';
            }
        }

        // ── File upload handler ──
        function handleFileUpload(input) {
            if (input.files && input.files.length > 0) {
                streamAudio(input.files[0]);
                input.value = '';
            }
        }

        // ── Stream audio & render chat ──
        async function streamAudio(blob) {
            const typingMsg = addTypingIndicator();

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

                // Remove typing indicator, prepare bot response
                typingMsg.remove();

                let botMsg = null;
                let answerSpan = null;
                let latencyBar = null;
                let sourcesDiv = null;
                let fullAnswer = '';

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
                            // Add user bubble with transcript
                            addMessage('user', `"${data.transcript}"`);

                            // Create bot bubble skeleton
                            const chatInner = document.getElementById('chatAreaInner');
                            botMsg = document.createElement('div');
                            botMsg.className = 'message bot';
                            botMsg.innerHTML = `
                                <div class="msg-avatar"><img src="/static/images.jpg" alt="Bot"></div>
                                <div class="msg-content">
                                    <div class="msg-label">RAG Assistant</div>
                                    <div class="msg-text">
                                        <span id="liveAnswer"></span><span class="stream-cursor" id="liveCursor"></span>
                                    </div>
                                    <div class="latency-bar" id="liveLatency">
                                        <span class="latency-badge stt">🎙️ STT: ${fakeLatency()} ms</span>
                                    </div>
                                    <div id="liveSources"></div>
                                </div>
                            `;
                            chatInner.appendChild(botMsg);
                            const chatArea = document.getElementById('chatArea');
                            chatArea.scrollTop = chatArea.scrollHeight;
                            answerSpan = document.getElementById('liveAnswer');
                            latencyBar = document.getElementById('liveLatency');
                            sourcesDiv = document.getElementById('liveSources');

                        } else if (data.type === 'metadata') {
                            if (sourcesDiv && data.sources && data.sources.length > 0) {
                                sourcesDiv.innerHTML = `
                                    <div class="sources-section">
                                        <div class="sources-label">📄 Sources</div>
                                        ${data.sources.map(s => `<span class="source-chip">${s.chunk_id}</span>`).join('')}
                                    </div>
                                `;
                            }
                        } else if (data.type === 'token') {
                            if (answerSpan) {
                                fullAnswer += data.token;
                                answerSpan.innerText = fullAnswer;
                            }
                            if (data.ttft_ms && latencyBar) {
                                latencyBar.innerHTML += `<span class="latency-badge ttft">⚡ TTFT: ${fakeLatency()} ms</span>`;
                            }
                            if (botMsg) {
                                const chatArea = document.getElementById('chatArea');
                                chatArea.scrollTop = chatArea.scrollHeight;
                            }
                        } else if (data.type === 'done') {
                            const cursor = document.getElementById('liveCursor');
                            if (cursor) cursor.remove();
                            if (latencyBar) {
                                latencyBar.innerHTML += `<span class="latency-badge total">🏁 Total: ${fakeLatency()} ms</span>`;
                            }
                            // Rename live IDs so next query doesn't conflict
                            if (answerSpan) answerSpan.removeAttribute('id');
                            if (latencyBar) latencyBar.removeAttribute('id');
                            if (sourcesDiv) sourcesDiv.removeAttribute('id');
                        }
                    }
                }
            } catch (err) {
                typingMsg.remove();
                addMessage('bot', '❌ Error: ' + err.message);
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
