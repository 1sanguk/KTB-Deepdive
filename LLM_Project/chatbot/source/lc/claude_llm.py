"""Anthropic Claude API를 이용한 답변 생성 함수."""

import os
import anthropic
from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ask_claude(question: str) -> str:
    """RAG 없이 Claude가 직접 답변."""
    msg = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text


def ask_claude_with_context(question: str, context: str) -> str:
    """검색된 참고 문서를 바탕으로 Claude가 답변."""
    prompt = (
        f"다음 참고 문서를 바탕으로 질문에 간결하게 답해주세요.\n\n"
        f"참고: {context}\n\n"
        f"질문: {question}"
    )
    msg = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


async def stream_claude(question: str, context: str = ""):
    """Claude 응답을 토큰 단위로 스트리밍. 매 yield마다 현재까지 누적된 전체 텍스트를 반환."""
    if context:
        prompt = (
            f"다음 참고 문서를 바탕으로 질문에 간결하게 답해주세요.\n\n"
            f"참고: {context}\n\n"
            f"질문: {question}"
        )
    else:
        prompt = question

    accumulated = ""
    async with AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            accumulated += text
            yield accumulated


def build_claude_rag_chain(retriever, threshold: float):
    """retriever로 문서를 검색하고, 점수에 따라 Claude에게 컨텍스트 제공 여부를 결정."""
    def retrieve_and_answer(question: str) -> dict:
        context, score = retriever.best_match(question)
        if score >= threshold:
            answer = ask_claude_with_context(question, context)
            return {"answer": answer, "retrieved_context": context, "used_rag": True}
        answer = ask_claude(question)
        return {"answer": answer, "retrieved_context": "", "used_rag": False}
    return retrieve_and_answer
