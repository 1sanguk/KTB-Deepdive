from typing import Callable, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from lg.nodes import (
    make_retriever_node,
    make_grade_node,
    make_generate_span_node,
    make_generate_direct_node,
    make_generate_claude_context_node,
    make_generate_claude_direct_node,
    make_claude_agent_node,
    make_tool_executor_node,
)
from lg.models import GraphState, AgentState
from lc.retriever import HybridRetriever
from lc.llm import SOP_GPT_LLM

# 최대 재시도 횟수: thresholds 길이 - 1 과 맞춰야 의미가 있다.
MAX_RETRIES = 2


def _init_graph_state(state: GraphState) -> dict:
    """user 메시지를 한 번만 messages에 추가하는 진입 노드 (retry 시 중복 방지)."""
    return {"messages": [{"role": "user", "content": state["query"]}]}


def _route(state: GraphState) -> str:
    """grade 노드 이후 분기 결정. used_rag=True라도 문서가 빈약하면 retry/direct로 돌린다."""
    if state['used_rag']:
        docs = state.get('documents', [])
        doc = docs[0].strip() if docs else ""
        if len(doc) > 20:
            return "span"
        return "retry" if state['retry_count'] < MAX_RETRIES else "direct"
    elif state['retry_count'] < MAX_RETRIES:
        return "retry"
    else:
        return "direct"




def increment_retry(state: GraphState) -> dict:
    """retry_count를 1 올린 뒤 retrieve 노드로 되돌아가기 위한 전용 노드."""
    return {"retry_count": state['retry_count'] + 1}


def build_graph(retriever: HybridRetriever, qa_llm: SOP_GPT_LLM, span_extractor_fn: Callable[[dict], str], threshold: list[float], checkpointer: Optional[MemorySaver] = None) -> CompiledStateGraph:
    """SOP_GPT 기반 LangGraph 파이프라인을 빌드해 컴파일된 그래프를 반환.

    흐름: init → retrieve → grade → (span | retry → retrieve | direct)
    checkpointer를 넘기면 thread_id 기반 대화 메모리가 활성화된다.
    """
    graph = StateGraph(GraphState)

    graph.add_node("init",            _init_graph_state)
    graph.add_node("retrieve",        make_retriever_node(retriever))
    graph.add_node("grade",           make_grade_node(threshold))
    graph.add_node("generate_span",   make_generate_span_node(span_extractor_fn))
    graph.add_node("generate_direct", make_generate_direct_node(qa_llm))
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("init")

    graph.add_edge("init",            "retrieve")
    graph.add_edge("retrieve",        "grade")
    graph.add_edge("increment_retry", "retrieve")

    graph.add_conditional_edges(
        "grade",
        _route,
        {
            "span":   "generate_span",
            "direct": "generate_direct",
            "retry":  "increment_retry",
        },
    )

    graph.add_edge("generate_span",   END)
    graph.add_edge("generate_direct", END)

    return graph.compile(checkpointer=checkpointer)


def build_claude_graph(retriever: HybridRetriever, threshold: list[float], checkpointer: Optional[MemorySaver] = None) -> CompiledStateGraph:
    """Claude 기반 LangGraph 파이프라인을 빌드해 컴파일된 그래프를 반환.

    SOP_GPT 그래프와 동일한 구조지만 generate 노드가 Claude API를 호출한다.
    """
    graph = StateGraph(GraphState)

    graph.add_node("init",            _init_graph_state)
    graph.add_node("retrieve",        make_retriever_node(retriever))
    graph.add_node("grade",           make_grade_node(threshold))
    graph.add_node("generate_span",   make_generate_claude_context_node())
    graph.add_node("generate_direct", make_generate_claude_direct_node())
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("init")

    graph.add_edge("init",            "retrieve")
    graph.add_edge("retrieve",        "grade")
    graph.add_edge("increment_retry", "retrieve")

    graph.add_conditional_edges(
        "grade",
        _route,
        {
            "span":   "generate_span",
            "direct": "generate_direct",
            "retry":  "increment_retry",
        },
    )

    graph.add_edge("generate_span",   END)
    graph.add_edge("generate_direct", END)

    return graph.compile(checkpointer=checkpointer)


def _agent_should_continue(state: AgentState) -> str:
    """tool_use면 tool_executor로, end_turn이면 END."""
    if state['stop_reason'] == "tool_use":
        return "tool"
    return "end"


def _init_messages(state: AgentState) -> AgentState:
    """query를 최초 user 메시지로 변환."""
    return {**state, "messages": [{"role": "user", "content": state['query']}]}


def build_claude_agent_graph(retriever: HybridRetriever) -> CompiledStateGraph:
    """Claude tool_use 기반 Agent 그래프.

    흐름: init → agent → (tool_use → tool_executor → agent | end_turn → END)
    """
    graph = StateGraph(AgentState)

    graph.add_node("init",          _init_messages)
    graph.add_node("agent",         make_claude_agent_node())
    graph.add_node("tool_executor", make_tool_executor_node(retriever))

    graph.set_entry_point("init")
    graph.add_edge("init", "agent")
    graph.add_conditional_edges(
        "agent",
        _agent_should_continue,
        {"tool": "tool_executor", "end": END},
    )
    graph.add_edge("tool_executor", "agent")

    return graph.compile()



