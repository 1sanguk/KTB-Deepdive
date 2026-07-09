from typing import Callable

from lg.models import GraphState, AgentState
from lc.retriever import HybridRetriever
from lc.chain import _expand_span
from llm.sop_llm import SOP_GPT_LLM
from llm.claude_llm import ask_claude, ask_claude_with_context, call_claude_agent_sync
from llm.qwen_llm import QwenBase


# ── SOP_GPT 노드 ───────────────────────────────────────────────────────────────

def make_retriever_node(retriever: HybridRetriever) -> Callable[[GraphState], GraphState]:
    """retriever를 클로저로 캡처해 그래프에 등록 가능한 retrieve 노드를 반환."""
    def retrieve(state: GraphState) -> dict:
        context, score = retriever.best_match(state['query'])
        return {"documents": [context], "score": score}

    return retrieve


def make_grade_node(thresholds: list[float]) -> Callable[[GraphState], GraphState]:
    """retry_count가 늘수록 임계값을 낮춰(완화) RAG 경로 진입 기회를 높이는 grade 노드를 반환."""
    def grade(state: GraphState) -> dict:
        idx = min(state['retry_count'], len(thresholds) - 1)
        return {"used_rag": state['score'] >= thresholds[idx]}

    return grade


def make_generate_span_node(span_extractor_fn: Callable[[dict], str]) -> Callable[[GraphState], GraphState]:
    """RAG 경로: span을 추출한 뒤 주변 문장까지 포함해 반환하는 노드를 반환."""
    def generate_span(state: GraphState) -> GraphState:
        context = state['documents'][0]
        span = span_extractor_fn({"question": state['query'], "context": context})
        answer = _expand_span(span, context)
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

    return generate_span


def make_generate_direct_node(qa_llm: SOP_GPT_LLM) -> Callable[[GraphState], GraphState]:
    """직접 답변 경로: 검색 없이 QA LLM 단독으로 답변하는 노드를 반환."""
    def generate_direct(state: GraphState) -> dict:
        answer = qa_llm.invoke(f"질문: {state['query']}\n답변: ")
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

    return generate_direct


# ── Claude 노드 ────────────────────────────────────────────────────────────────

def make_generate_claude_context_node() -> Callable[[GraphState], GraphState]:
    """RAG 경로: 검색 문서를 컨텍스트로 Claude에 질문하는 노드를 반환."""
    def generate_context(state: GraphState) -> dict:
        answer = ask_claude_with_context(state['query'], state['documents'][0])
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

    return generate_context


def make_generate_claude_direct_node() -> Callable[[GraphState], GraphState]:
    """직접 답변 경로: 문서 없이 Claude에 질문하는 노드를 반환."""
    def generate_direct(state: GraphState) -> dict:
        answer = ask_claude(state['query'])
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

    return generate_direct


def make_claude_agent_node() -> Callable[[AgentState], AgentState]:
    def agent(state: AgentState) -> AgentState:
        response = call_claude_agent_sync(state['messages'])
        if isinstance(response, str):
            return {
                **state, "answer": response, "stop_reason": "end_turn"
            }
        
        new_messages = state['messages'] + [{"role": "assistant", "content": response.content}]
        answer = ""
        if response.stop_reason in ("end_turn", "max_tokens"):
            for block in response.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
                    answer = block.text
                    break
            if not answer:
                for msg in reversed(new_messages):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str) and content:
                            answer = content
                            break
        return {
            **state, "messages": new_messages, "stop_reason": response.stop_reason, "answer": answer,
        }
    return agent


def make_tool_executor_node(retriever: HybridRetriever) -> Callable[[AgentState], AgentState]:
    def tool_executor(state: AgentState) -> AgentState:
        last_content = state['messages'][-1]['content']
        tool_results = []
        retrieved_context = state['retrieved_context']
        used_rag = state['used_rag']

        for block in last_content:
            if hasattr(block, "type") and block.type == "tool_use":
                if block.name == "search_knowledge_base":
                    query = block.input.get("query", state['query'])
                    context, _ = retriever.best_match(query)
                    retrieved_context = context
                    used_rag = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": context,
                    })

        new_messages = state['messages'] + [{"role": "user", "content": tool_results}]
        return {
            **state, "retrieved_context": retrieved_context, "used_rag": used_rag, "messages": new_messages
        }
    
    return tool_executor


# ── Qwen 노드 ──────────────────────────────────────────────────────────────────

def make_generate_qwen_context_node(qwen_llm: QwenBase) -> Callable[[GraphState], GraphState]:
    """RAG 경로: 검색 문서를 컨텍스트로 Qwen에 질문하는 노드를 반환."""
    def generate_context(state: GraphState) -> dict:
        answer = qwen_llm.ask_with_context(state['query'], state['documents'][0])
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}
    return generate_context


def make_generate_qwen_direct_node(qwen_llm: QwenBase) -> Callable[[GraphState], GraphState]:
    """직접 답변 경로: 문서 없이 Qwen에 질문하는 노드를 반환."""
    def generate_direct(state: GraphState) -> dict:
        answer = qwen_llm.ask(state['query'])
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}
    return generate_direct
