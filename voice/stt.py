"""
Speech-to-Text (STT) Module — Sarvam AI Dedicated

PURPOSE:
    Transcribes audio using Sarvam AI Saaras v3 model (fast Indic & English speech-to-text).
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from config.settings import settings

logger = logging.getLogger(__name__)


class SarvamSTT:
    """
    Dedicated Sarvam AI Speech-to-Text client.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        model: str = "saaras:v3",
    ) -> tuple[str, float]:
        """
        Transcribe raw audio bytes into text using Sarvam AI.

        Endpoint: https://api.sarvam.ai/speech-to-text
        Supported models: 'saaras:v3', 'saarika:v2.5', 'saarika:flash'
        """
        t0 = time.perf_counter()

        # Dynamic reload from .env in case user updated it recently
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
        key = self.api_key or os.getenv("SARVAM_API_KEY") or settings.sarvam_api_key

        if not key or key.strip() in ("", "your_sarvam_api_key_here"):
            raise ValueError(
                "SARVAM_API_KEY is not set or empty in .env. Please add your key: SARVAM_API_KEY=your_key"
            )

        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": key.strip().strip('"'),
        }
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "model": model,
            "language_code": language_code or settings.sarvam_language_code,
        }

        logger.info("Sending audio to Sarvam AI STT (model: %s, language: %s)...", model, data["language_code"])
        response = requests.post(url, headers=headers, files=files, data=data, timeout=15)

        if response.status_code != 200:
            logger.error("Sarvam AI STT error [%d]: %s", response.status_code, response.text)
            response.raise_for_status()

        res_json = response.json()
        transcript = res_json.get("transcript", "").strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Sarvam STT completed in %.2f ms: '%s'", elapsed_ms, transcript)
        return transcript, elapsed_ms


def get_stt_client() -> SarvamSTT:
    """Return dedicated Sarvam STT client."""
    return SarvamSTT()
