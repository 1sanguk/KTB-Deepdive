"""SSE(Server-Sent Events) 스트리밍 헬퍼."""

import asyncio
import json
import queue
import re
import threading
import uuid
import state as _state
from typing import Any, AsyncGenerator, Callable

from history import append_compare_turn, load_history
from lc.chain import _expand_span
from lc.router import classify_question, CHAIN_FOR, MODE_LABEL
from llm.claude_llm import stream_claude
from llm.gemini_llm import stream_gemini
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
            _state.append_history(thread_id, [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ])
        except Exception:
            pass


MODEL_NAMES_KO = {
    "sop":    "SOP_GPT",
    "claude": "Claude (Haiku)",
    "qwen":   "Qwen3-1.7B (BF16)",
    "qwen-q": "Qwen3-1.7B (Q4)",
}


def _classify_gemini_error(text: str) -> str:
    if "503" in text or "UNAVAILABLE" in text:
        return "Gemini 서버 일시 과부하 (503)"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "API 사용량 한도 초과 (429)"
    if "401" in text or "API_KEY" in text:
        return "API 인증 오류 (401)"
    if "재시도" in text:
        return "Gemini 서버 과부하 — 재시도 후에도 응답 없음"
    return "Gemini 연결 오류"


def _pick_fallback_model(model_answers: dict) -> str:
    """Gemini 실패 시 오류 없는 답변 중 가장 긴 모델을 선택."""
    valid = {k: v for k, v in model_answers.items() if v and not v.startswith("[")}
    if not valid:
        return "claude"
    return max(valid, key=lambda k: len(valid[k]))


# 텍스트 전체에서 모델 키워드 추론 (우선순위 높은 것 먼저)
_MODEL_KEYWORDS = [
    ("qwen-q", ["qwen q4", "qwen-q", "q4_k", "양자화 qwen"]),
    ("qwen",   ["qwen bf16", "qwen bf", "bf16"]),
    ("claude", ["claude"]),
    ("sop",    ["sop_gpt", "sop gpt"]),
]


def _parse_best_model(text: str) -> str | None:
    """judge 텍스트에서 최고 모델 키를 추출. 실패 시 None."""
    # 1차: 명시적 패턴 (공백·대괄호 허용)
    m = re.search(r'최선\s*모델[:\s]+\[?([\w-]+)', text)
    if m:
        candidate = m.group(1).strip().lower()
        if candidate in ("sop", "claude", "qwen", "qwen-q"):
            return candidate
    # 2차: 마지막 200자에서 키워드 추론
    tail = text[-200:].lower()
    for key, keywords in _MODEL_KEYWORDS:
        if any(kw in tail for kw in keywords):
            return key
    return None


async def compare_stream(question: str, thread_id: str | None) -> AsyncGenerator[str, None]:
    """4개 모델을 병렬 실행하고 Claude가 최선 답변을 판별하는 SSE 스트림."""
    model_answers: dict = {}
    merged_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def run_via_lg(key: str, graph: Any):
        acc = ""
        try:
            if graph is None:
                acc = "[모델 없음]"
                await merged_q.put({"type": "model_text", "model": key, "text": acc})
                return

            suffix = {"sop": "", "claude": ":c", "qwen": ":bf16", "qwen-q": ":q4"}[key]
            base = thread_id or str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": base + suffix}}

            q = queue.Queue()

            def _fn():
                try:
                    for chunk in graph.stream({
                        "query":       question,
                        "documents":   [],
                        "score":       0.0,
                        "answer":      "",
                        "used_rag":    False,
                        "retry_count": 0,
                        "messages":    [],
                    }, config=cfg):
                        q.put(chunk)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(None)

            threading.Thread(target=_fn, daemon=True).start()

            while True:
                chunk = await loop.run_in_executor(None, q.get)
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    acc = f"[오류] {type(chunk).__name__}: {chunk}"
                    await merged_q.put({"type": "model_text", "model": key, "text": acc})
                    break
                if "generate_span" in chunk:
                    acc = chunk["generate_span"].get("answer", "")
                    await merged_q.put({"type": "model_text", "model": key, "text": acc})
                elif "generate_direct" in chunk:
                    acc = chunk["generate_direct"].get("answer", "")
                    await merged_q.put({"type": "model_text", "model": key, "text": acc})
        except Exception as e:
            acc = f"[오류] {e}"
            await merged_q.put({"type": "model_text", "model": key, "text": acc})
        finally:
            if not acc:
                acc = "응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요."
                await merged_q.put({"type": "model_text", "model": key, "text": acc})
            model_answers[key] = acc
            await merged_q.put({"type": "model_done", "model": key})

    tasks = [
        asyncio.create_task(run_via_lg("sop",    _state.lg_graph)),
        asyncio.create_task(run_via_lg("claude", _state.claude_graph)),
        asyncio.create_task(run_via_lg("qwen",   _state.qwen_graph)),
        asyncio.create_task(run_via_lg("qwen-q", _state.qwen_quant_graph)),
    ]

    done_count = 0
    while done_count < 4:
        event = await merged_q.get()
        yield _sse(event)
        if event["type"] == "model_done":
            done_count += 1

    await asyncio.gather(*tasks, return_exceptions=True)

    # Gemini가 4개 답변 중 최선을 판별 (자기 편향 방지)
    judge_prompt = (
        f"다음은 동일한 질문에 대한 4개 AI 모델의 답변입니다.\n\n"
        f"질문: {question}\n\n"
        f"- SOP_GPT: {model_answers.get('sop', '[없음]')}\n"
        f"- Claude: {model_answers.get('claude', '[없음]')}\n"
        f"- Qwen BF16: {model_answers.get('qwen', '[없음]')}\n"
        f"- Qwen Q4: {model_answers.get('qwen-q', '[없음]')}\n\n"
        f"질문에 대한 답변으로 가장 정확하고 유용한 답변을 선택하고 이유를 2~3문장으로 설명하세요."
        f"첫 줄에 반드시 아래 형식 중 하나를 그대로 출력하세요 (다른 문자 없이):\n"
        f"최선모델: sop\n최선모델: claude\n최선모델: qwen\n최선모델: qwen-q"
    )

    acc_judge = ""
    try:
        async for text in stream_gemini(judge_prompt):
            acc_judge = text
            yield _sse({"type": "judge_text", "text": text})
    except Exception as e:
        yield _sse({"type": "judge_text", "text": f"[판별 오류] {e}"})

    best_model = "claude"
    judge_error = None
    parsed = _parse_best_model(acc_judge)
    if parsed:
        best_model = parsed
    else:
        judge_error = _classify_gemini_error(acc_judge)
        best_model = _pick_fallback_model(model_answers)

    evt = {
        "type": "judge_done",
        "best_model": best_model,
        "best_text": model_answers.get(best_model, ""),
    }
    if judge_error:
        evt["judge_error"] = judge_error
        evt["judge_fallback_model"] = MODEL_NAMES_KO.get(best_model, best_model)

    yield _sse(evt)

    if thread_id:
        try:
            append_compare_turn(thread_id, question, model_answers, best_model)
        except Exception:
            pass

    yield _sse({"type": "done"})


async def session_judge_stream(thread_id: str | None) -> AsyncGenerator[str, None]:
    """세션 전체 4모델 답변을 Claude가 종합 평가하는 SSE 스트림."""
    if not thread_id:
        yield _sse({"type": "judge_text", "text": "세션 ID가 없습니다."})
        yield _sse({"type": "done"})
        return

    messages = load_history(thread_id)

    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            q_text = msg.get("content", "")
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                a_msg = messages[i + 1]
                parts = [f"Q: {q_text}"]
                if "models" in a_msg:
                    for k, v in a_msg["models"].items():
                        parts.append(f"  [{k}]: {v}")
                else:
                    parts.append(f"  [답변]: {a_msg.get('content', '')}")
                turns.append("\n".join(parts))
                i += 2
                continue
        i += 1

    if not turns:
        yield _sse({"type": "judge_text", "text": "평가할 대화 기록이 없습니다."})
        yield _sse({"type": "done"})
        return

    prompt = (
        "다음은 AI 챗봇 4개 모델(SOP_GPT, Claude, Qwen BF16, Qwen Q4)의 세션 대화 기록입니다.\n\n"
        + "\n\n".join(turns)
        + "\n\n각 모델별로 이 세션에서의 답변 품질을 평가해주세요. "
        "정확성, 완성도, 한국어 자연스러움을 기준으로 강점과 약점을 설명하고, "
        "최종적으로 이 세션에서 가장 잘 답변한 모델을 선정해 주세요."
    )

    try:
        async for text in stream_gemini(prompt, max_tokens=None):
            yield _sse({"type": "judge_text", "text": text})
    except Exception as e:
        yield _sse({"type": "judge_text", "text": f"[오류] {e}"})

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


async def auto_sop_stream(question: str, thread_id: str | None) -> AsyncGenerator[str, None]:
    """질문을 자동 분류해 적합한 SOP_GPT 체인으로 라우팅."""
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
