"""SSE(Server-Sent Events) 스트리밍 헬퍼."""

import asyncio
import json
import queue
import threading
from typing import Any, AsyncGenerator, Callable

from lc.chain import _expand_span
from llm.claude_llm import stream_claude
from llm.sop_llm import SOP_GPT_LLM
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

async def sop_lg_stream(graph: CompiledStateGraph, question: str, thread_id: str | None = None) -> AsyncGenerator[str, None]:
    import uuid
    import state as _state
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
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
                'messages': [],
            }, config=cfg):
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
            node_state = chunk['generate_span']
            docs = node_state.get('documents', [])
            if docs:
                yield _sse({"type": "rag_context", "text": docs[0]})
            yield _sse({"type": "text", "text": node_state['answer']})
        elif "generate_direct" in chunk:
            yield _sse({"type": "text_fallback", "text": chunk['generate_direct']['answer']})

    yield _sse({"type": "done"})

    # 스트림 완료 후 JSON 파일에 히스토리 저장
    if thread_id:
        try:
            snap = graph.get_state(cfg)
            if snap and snap.values:
                _state.save_history(thread_id, snap.values.get("messages", []))
        except Exception:
            pass
    

async def with_history(gen: AsyncGenerator[str, None], thread_id: str | None, question: str) -> AsyncGenerator[str, None]:
    """스트림을 그대로 통과시키되, 완료 후 question+answer를 JSON 파일에 추가."""
    import state as _state
    answer = ""
    async for chunk in gen:
        yield chunk
        if not thread_id:
            continue
        try:
            data = json.loads(chunk.removeprefix("data: ").strip())
            if data.get("type") in ("text", "text_fallback"):
                answer = data.get("text", "")
        except Exception:
            pass
    if thread_id and answer:
        _state.append_history(thread_id, [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])


async def agent_lg_stream(graph: CompiledStateGraph, question: str, thread_id: str | None = None) -> AsyncGenerator[str, None]:
    """Claude Agent Graph(AgentState) 전용 스트리밍."""
    q = queue.Queue()

    def run():
        try:
            for chunk in graph.stream({
                "query":             question,
                "messages":          [],
                "stop_reason":       "",
                "answer":            "",
                "used_rag":          False,
                "retrieved_context": "",
            }):
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()
    answer = ""
    while True:
        chunk = await loop.run_in_executor(None, q.get)
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            yield _sse({"type": "text", "text": f"[오류] Agent 실행 중 오류가 발생했습니다. ({type(chunk).__name__})"})
            break

        if "tool_executor" in chunk:
            context = chunk["tool_executor"].get("retrieved_context", "")
            yield _sse({"type": "status", "text": "검색 중...."})
            if context:
                yield _sse({"type": "rag_context", "text": context})
        elif "agent" in chunk:
            node_state = chunk["agent"]
            if node_state.get("stop_reason") in ("end_turn", "max_tokens") and node_state.get("answer"):
                answer = node_state["answer"]
                yield _sse({"type": "text", "text": answer})

    yield _sse({"type": "done"})

    if thread_id and answer:
        try:
            import state as _state
            _state.append_history(thread_id, [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ])
        except Exception:
            pass


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


async def auto_sop_stream(question: str, thread_id: str | None) -> AsyncGenerator[str, None]:
    """질문을 자동 분류해 적합한 SOP_GPT 체인으로 라우팅."""
    from lc.router import classify_question, CHAIN_FOR, MODE_LABEL
    import state as _state

    label = classify_question(question)
    mode  = CHAIN_FOR[label]
    yield _sse({"type": "mode", "text": mode, "label": MODE_LABEL[mode]})

    if mode == "basic":
        async for chunk in with_history(
            sop_stream(_state.qa_llm, f"질문: {question}\n답변: "),
            thread_id, question,
        ):
            yield chunk
    elif mode == "langchain":
        async for chunk in with_history(
            sop_rag_stream(
                _state.lc_retriever, _state.qa_llm,
                _state.span_extractor_fn, _state.RAG_SIM_THRESHOLD,
                question,
            ),
            thread_id, question,
        ):
            yield chunk
    elif mode == "langgraph":
        async for chunk in sop_lg_stream(_state.lg_graph, question, thread_id=thread_id):
            yield chunk


async def auto_claude_stream(question: str, thread_id: str | None) -> AsyncGenerator[str, None]:
    """질문을 자동 분류해 적합한 Claude 체인으로 라우팅."""
    from lc.router import classify_question, CHAIN_FOR, MODE_LABEL
    import state as _state

    label = classify_question(question)
    mode  = CHAIN_FOR[label]
    yield _sse({"type": "mode", "text": mode, "label": MODE_LABEL[mode]})

    claude_tid = (thread_id + ":c") if thread_id else None

    if mode == "basic":
        async def _gen():
            async for text in stream_claude(question):
                yield _sse({"type": "text", "text": text})
            yield _sse({"type": "done"})
        async for chunk in with_history(_gen(), claude_tid, question):
            yield chunk
    elif mode == "langchain":
        async for chunk in with_history(
            claude_rag_stream(_state.lc_retriever, _state.RAG_SIM_THRESHOLD, question),
            claude_tid, question,
        ):
            yield chunk
    elif mode == "langgraph":
        async for chunk in agent_lg_stream(_state.claude_agent_graph, question, thread_id=claude_tid):
            yield chunk
