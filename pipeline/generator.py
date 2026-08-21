"""
Structured LLM Generator with Streaming Support

PURPOSE:
    Generates answers from retrieved contexts with structured JSON output, bulletproof fallback parsing,
    and real-time token streaming for sub-80ms Time-To-First-Token (TTFT).
"""

import json
import logging
import re
import time
from typing import Generator, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class StructuredGenerator:
    """
    LLM generator producing validated structured output with streaming capabilities.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _get_llm_client(self):
        if self.llm_client is None:
            from groq import Groq
            self.llm_client = Groq(api_key=settings.groq_api_key)
        return self.llm_client

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        formatted_context = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        return f"""You are a precise, grounded assistant. Answer the user's question ONLY using the provided context.

Context:
{formatted_context}

Question:
{query}

Instructions:
1. Answer ONLY using facts directly stated in the context. Do NOT use external knowledge.
2. If the context does not contain sufficient facts to answer the question, state: "Content not found."
3. Keep the answer extremely concise (maximum 35 words).
4. Provide a 1-sentence summary (maximum 15 words).
5. Output ONLY valid JSON in this exact structure without thinking process or code blocks:
{{
    "answer": "<your answer>",
    "summary": "<one sentence summary>",
    "grounded": true or false
}}"""

    def _parse_response(self, text: Optional[str]) -> Optional[dict]:
        """Bulletproof extraction of answer and summary from LLM output."""
        if not text:
            return None

        text = text.strip()

        # 1. Strip reasoning / think tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "answer" in data:
                return {
                    "answer": str(data["answer"]),
                    "summary": str(data.get("summary", "")),
                    "grounded": bool(data.get("grounded", True)),
                }
        except Exception:
            pass

        # 3. Extract from markdown code fence ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "answer" in data:
                    return {
                        "answer": str(data["answer"]),
                        "summary": str(data.get("summary", "")),
                        "grounded": bool(data.get("grounded", True)),
                    }
            except Exception:
                pass

        # 4. Regex extraction for answer and summary fields
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)', text, re.DOTALL)
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)', text, re.DOTALL)
        if answer_match and answer_match.group(1).strip():
            ans = answer_match.group(1).replace('\\"', '"').strip()
            summ = summary_match.group(1).replace('\\"', '"').strip() if summary_match else ""
            return {
                "answer": ans,
                "summary": summ or (ans[:80] + "..." if len(ans) > 80 else ans),
                "grounded": True,
            }

        # 5. Fallback: Clean up any JSON brackets and use the raw text as answer
        clean_text = re.sub(r'[{}\[\]"]', "", text).strip()
        clean_text = re.sub(r"^(answer|summary)\s*:\s*", "", clean_text, flags=re.IGNORECASE).strip()
        if len(clean_text) > 0:
            return {
                "answer": clean_text,
                "summary": clean_text[:80] + "..." if len(clean_text) > 80 else clean_text,
                "grounded": True,
            }

        return None

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        retry_count: int = 1,
    ) -> tuple[dict, float]:
        """Generate structured answer synchronously."""
        t0 = time.perf_counter()
        prompt = self._build_prompt(query, context_chunks)
        client = self._get_llm_client()

        for attempt in range(retry_count + 1):
            try:
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                raw_text = response.choices[0].message.content
                parsed = self._parse_response(raw_text)

                if parsed is not None:
                    elapsed = (time.perf_counter() - t0) * 1000
                    return parsed, elapsed

                logger.warning("Attempt %d: Raw LLM output could not be parsed: %r", attempt + 1, raw_text)
            except Exception as e:
                logger.warning("Attempt %d: LLM call failed (%s)", attempt + 1, e)

        # Safe fallback if all attempts fail
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("Structured generation failed after %d attempts. Using safe fallback.", retry_count + 1)
        return {
            "answer": "Content not found in context.",
            "summary": "No relevant information available.",
            "grounded": False,
        }, elapsed

    def generate_stream(
        self,
        query: str,
        context_chunks: list[str],
    ) -> Generator[tuple[str, Optional[float], bool], None, None]:
        """
        Stream generated tokens in real time.

        Yields:
            (token_text, ttft_ms, is_finished)
        """
        t0 = time.perf_counter()
        formatted_context = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        stream_prompt = f"""You are a concise, helpful assistant. Answer the question in 1-2 direct sentences using ONLY the provided context.

Context:
{formatted_context}

Question:
{query}

Direct Answer:"""

        client = self._get_llm_client()
        first_token = True
        ttft_ms = None

        try:
            stream = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": stream_prompt}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=True,
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    if first_token:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        first_token = False
                        yield token, ttft_ms, False
                    else:
                        yield token, None, False

            yield "", None, True

        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            yield f" [Error: {e}]", None, True
