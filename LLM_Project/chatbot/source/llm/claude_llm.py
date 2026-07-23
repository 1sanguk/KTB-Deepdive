"""Anthropic Claude API를 이용한 답변 생성 함수."""

import asyncio
import logging
import os
from typing import AsyncGenerator, Callable

import anthropic
from anthropic import AsyncAnthropic
from lc.retriever import HybridRetriever

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512
_STREAM_MAX_RETRIES = 2
_STREAM_RETRY_DELAY = 2.0  # seconds


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _error_message(e: Exception) -> str:
    if isinstance(e, anthropic.AuthenticationError):
        return "[오류] API 키가 유효하지 않습니다."
    if isinstance(e, anthropic.RateLimitError):
        return "[오류] 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, anthropic.APIStatusError) and e.status_code == 402:
        return "[오류] Claude API 크레딧이 소진됐습니다. Anthropic Console에서 충전해주세요."
    if isinstance(e, anthropic.APIConnectionError):
        return "[오류] Claude API 서버에 연결할 수 없습니다."
    if isinstance(e, anthropic.APIStatusError):
        return f"[오류] Claude API 오류가 발생했습니다. (HTTP {e.status_code})"
    return f"[오류] 알 수 없는 오류가 발생했습니다. ({e})"


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, anthropic.APIStatusError):
        return e.status_code in (429, 503, 529)
    if isinstance(e, anthropic.APIConnectionError):
        return True
    return False


def ask_claude(question: str) -> str:
    """RAG 없이 Claude가 직접 답변."""
    try:
        msg = _client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text
    except Exception as e:
        return _error_message(e)


def ask_claude_with_context(question: str, context: str) -> str:
    """검색된 참고 문서를 바탕으로 Claude가 답변."""
    prompt = (
        f"다음 참고 문서를 참고하여 질문에 간결하게 답해주세요. "
        f"문서에 직접적인 내용이 없으면 일반 지식을 바탕으로 답변하세요.\n\n"
        f"참고: {context}\n\n"
        f"질문: {question}"
    )
    try:
        msg = _client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return _error_message(e)


async def stream_claude(question: str, context: str = "") -> AsyncGenerator[str, None]:
    """Claude 응답을 토큰 단위로 스트리밍.
    매 yield마다 현재까지 누적된 전체 텍스트를 반환.
    과부하(529/503)·빈 응답은 최대 2회 재시도.
    """
    if context:
        prompt = (
            f"다음 참고 문서를 참고하여 질문에 간결하게 답해주세요. "
            f"문서에 직접적인 내용이 없으면 일반 지식을 바탕으로 답변하세요.\n\n"
            f"참고: {context}\n\n"
            f"질문: {question}"
        )
    else:
        prompt = question

    for attempt in range(_STREAM_MAX_RETRIES):
        try:
            accumulated = ""
            async with AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    accumulated += text
                    yield accumulated

            if accumulated:
                return

            # HTTP 200이지만 토큰이 0개인 빈 응답
            logger.warning("Claude 빈 응답 (시도 %d/%d)", attempt + 1, _STREAM_MAX_RETRIES)
            if attempt < _STREAM_MAX_RETRIES - 1:
                yield f"[응답 없음, {int(_STREAM_RETRY_DELAY)}초 후 재시도 중...]"
                await asyncio.sleep(_STREAM_RETRY_DELAY)
            else:
                yield "응답을 받지 못했습니다. 잠시 후 다시 시도하거나 관리자에게 문의해 주세요."
                return

        except Exception as e:
            logger.error("Claude 스트리밍 오류 (시도 %d/%d): %s", attempt + 1, _STREAM_MAX_RETRIES, e)
            if _is_retryable(e) and attempt < _STREAM_MAX_RETRIES - 1:
                wait = _STREAM_RETRY_DELAY * (attempt + 1)
                yield f"[서버 과부하, {int(wait)}초 후 재시도 중...]"
                await asyncio.sleep(wait)
            else:
                yield _error_message(e)
                return


def build_claude_rag_chain(retriever: HybridRetriever, threshold: float) -> Callable[[str], dict]:
    """retriever로 문서를 검색하고, 점수에 따라 Claude에게 컨텍스트 제공 여부를 결정."""
    def retrieve_and_answer(question: str) -> dict:
        context, score = retriever.best_match(question)
        if score >= threshold:
            answer = ask_claude_with_context(question, context)
            return {"answer": answer, "retrieved_context": context, "used_rag": True}
        answer = ask_claude(question)
        return {"answer": answer, "retrieved_context": "", "used_rag": False}
    return retrieve_and_answer



AGENT_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "한국어 지식 베이스(KorQuAD 기반) 및 ragdata를 검색합니다. "
            "역사, 지리, 과학, 문화, 인물 등 사실 정보 질문에 사용하세요. "
            "일상 대화나 의견·추론 질문은 도구 없이 직접 답변하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 질문 (한국어)"}
            },
            "required": ["query"]
        }
    }
]

AGENT_SYSTEM = (
    "당신은 한국어 질의응답 도우미입니다.\n"
    "역사, 지리, 과학, 문화, 인물 등 사실 확인이 필요한 질문은 "
    "search_knowledge_base 도구로 지식 베이스를 검색하세요.\n"
    "일상 대화, 추론, 의견처럼 사실 검색이 불필요한 경우엔 도구 없이 직접 답변하세요.\n"
    "답변은 한국어로 간결하게 작성하세요."
)

def call_claude_agent_sync(messages: list) -> object:
    try:
        return _client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=AGENT_SYSTEM,
            tools=AGENT_TOOLS,
            messages=messages,
        )
    except Exception as e:
        return _error_message(e)
