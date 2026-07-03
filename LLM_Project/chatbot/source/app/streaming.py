"""SSE(Server-Sent Events) 스트리밍 헬퍼."""

import asyncio
import json
import queue
import threading
from typing import Any, AsyncGenerator, Callable

from lc.chain import _expand_span
from lc.claude_llm import stream_claude
from lc.llm import SOP_GPT_LLM
from langgraph.graph.state import CompiledStateGraph


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def sop_stream(llm: SOP_GPT_LLM, prompt: str) -> AsyncGenerator[str, None]:
    """동기 SOP_GPT 제너레이터를 스레드에서 실행해 SSE 이벤트를 yield."""
    q = queue.Queue()

    def run():
        try:
            for text in llm.stream_tokens(prompt):
                q.put(text)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        text = await loop.run_in_executor(None, q.get)
        if text is None:
            break
        yield _sse({"type": "text", "text": text})
    yield _sse({"type": "done"})


async def sop_rag_stream(
    retriever: Any,
    llm: SOP_GPT_LLM,
    span_extractor_fn: Callable[[dict], str],
    threshold: float,
    question: str,
) -> AsyncGenerator[str, None]:
    try:
        context, score = retriever.best_match(question)
        if score >= threshold:
            span = span_extractor_fn({"question": question, "context": context})
            answer = _expand_span(span, context)
            yield _sse({"type": "rag_context", "text": context})
            yield _sse({"type": "text", "text": answer})
            yield _sse({"type": "done"})
        else:
            async for chunk in sop_stream(llm, f"질문: {question}\n답변: "):
                yield chunk
    except Exception as e:
        yield _sse({"type": "text", "text": f"[오류] 처리 중 오류가 발생했습니다. ({type(e).__name__})"})
        yield _sse({"type": "done"})

async def sop_lg_stream(graph: CompiledStateGraph, question: str) -> AsyncGenerator[str, None]:
    q = queue.Queue()

    def run():
        try:
            for chunk in graph.stream({
                'query': question,
                'documents': [],
                'score': 0.0,
                'answer': '',
                'used_rag': False,
                'retry_count': 0,
            }):
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        chunk = await loop.run_in_executor(None, q.get)
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            yield _sse({"type": "text", "text": f"[오류] 파이프라인 실행 중 오류가 발생했습니다. ({type(chunk).__name__})"})
            break

        if "retrieve" in chunk:
            yield _sse({"type": "status", "text": "검색 중...."})
        elif "increment_retry" in chunk:
            yield _sse({"type": "status", "text": f"재시도 중.... ({chunk['increment_retry']['retry_count']})"})
        elif "generate_span" in chunk:
            state = chunk['generate_span']
            if state['documents']:
                yield _sse({"type": "rag_context", "text": state['documents'][0]})
            yield _sse({"type": "text", "text": state['answer']})
        elif "generate_direct" in chunk:
            yield _sse({"type": "text_fallback", "text": chunk['generate_direct']['answer']})

    yield _sse({"type": "done"})
    

async def claude_rag_stream(retriever: Any, threshold: float, question: str) -> AsyncGenerator[str, None]:
    context, score = retriever.best_match(question)
    if score >= threshold:
        yield _sse({"type": "rag_context", "text": context})
        async for text in stream_claude(question, context):
            yield _sse({"type": "text", "text": text})
    else:
        async for text in stream_claude(question):
            yield _sse({"type": "text", "text": text})
    yield _sse({"type": "done"})
