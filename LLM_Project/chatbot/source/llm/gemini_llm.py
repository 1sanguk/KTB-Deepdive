"""Google Gemini API를 이용한 판별(Judge) 스트리밍 함수."""

import asyncio
import logging
import os
import re
from typing import AsyncGenerator

from google import genai

logger = logging.getLogger(__name__)

MODEL = "gemini-flash-latest"
_MAX_RETRIES = 3
_RETRY_DELAY = 5.0  # seconds

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    return _client


def _is_retryable(e: Exception) -> bool:
    msg = str(e)
    return (
        "503" in msg or "UNAVAILABLE" in msg
        or "502" in msg or "Bad Gateway" in msg
        or "429" in msg or "RESOURCE_EXHAUSTED" in msg
    )


def _extract_code(e: Exception) -> str:
    m = re.search(r'"code":\s*(\d+)', str(e))
    if m:
        return m.group(1)
    m = re.search(r'\b(4\d\d|5\d\d)\b', str(e))
    return m.group(1) if m else "알 수 없음"


async def stream_gemini(prompt: str, max_tokens: int | None = 1024) -> AsyncGenerator[str, None]:
    """Gemini 응답을 스트리밍. 503/429 일시 오류는 최대 3회 재시도."""
    client = _get_client()
    config = {"max_output_tokens": max_tokens} if max_tokens is not None else {}

    for attempt in range(_MAX_RETRIES):
        try:
            accumulated = ""
            async for chunk in await client.aio.models.generate_content_stream(
                model=MODEL,
                contents=prompt,
                config=config,
            ):
                if chunk.text:
                    accumulated += chunk.text
                    yield accumulated
            return
        except Exception as e:
            code = _extract_code(e)
            logger.error("Gemini API 오류 (시도 %d/%d) [%s]: %s", attempt + 1, _MAX_RETRIES, code, e)
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                yield f"[{int(wait)}초 후 재시도 중... ({attempt + 1}/{_MAX_RETRIES - 1})]"
                await asyncio.sleep(wait)
            else:
                yield f"오류 {code}가 발생했습니다. 잠시 후 다시 시도하거나 관리자에게 문의해 주세요."
                return
