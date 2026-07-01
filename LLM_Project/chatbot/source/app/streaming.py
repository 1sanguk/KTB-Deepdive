"""SSE(Server-Sent Events) 스트리밍 헬퍼."""

import asyncio
import json
import queue
import threading

from lc.claude_llm import stream_claude


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def sop_stream(llm, prompt: str):
    """동기 SOP_GPT 제너레이터를 스레드에서 실행해 SSE 이벤트를 yield."""
    q: queue.Queue = queue.Queue()

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


async def sop_rag_stream(retriever, llm, span_extractor_fn, threshold: float, question: str):
    context, score = retriever.best_match(question)
    if score >= threshold:
        answer = span_extractor_fn({"question": question, "context": context})
        yield _sse({"type": "rag_context", "text": context})
        yield _sse({"type": "text", "text": answer})
        yield _sse({"type": "done"})
    else:
        async for chunk in sop_stream(llm, f"질문: {question}\n답변: "):
            yield chunk


async def claude_rag_stream(retriever, threshold: float, question: str):
    context, score = retriever.best_match(question)
    if score >= threshold:
        yield _sse({"type": "rag_context", "text": context})
        async for text in stream_claude(question, context):
            yield _sse({"type": "text", "text": text})
    else:
        async for text in stream_claude(question):
            yield _sse({"type": "text", "text": text})
    yield _sse({"type": "done"})
