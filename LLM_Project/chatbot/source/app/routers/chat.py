from fastapi import APIRouter

import state
from lc.claude_llm import ask_claude
from models import ChatRequest, ChatResponse, GenerateRequest, GenerateResponse

router = APIRouter()

_ERR_SOP  = "[오류] 모델 추론 중 오류가 발생했습니다."
_ERR_RAG  = "[오류] 검색 중 오류가 발생했습니다."
_ERR_GRAPH = "[오류] 파이프라인 실행 중 오류가 발생했습니다."


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        text = state.gen_llm.invoke(req.prompt)
    except Exception:
        text = _ERR_SOP
    return GenerateResponse(text=text)


@router.post("/chat/basic", response_model=ChatResponse)
def chat_basic(req: ChatRequest) -> ChatResponse:
    try:
        answer = state.basic_chain.invoke(req.question)
    except Exception:
        answer = _ERR_SOP
    return ChatResponse(answer=answer, retrieved_context="", used_rag=False)


@router.post("/chat/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest) -> ChatResponse:
    try:
        result = state.tfidf_rag_chain.invoke(req.question)
    except Exception:
        result = {"answer": _ERR_RAG, "retrieved_context": "", "used_rag": False}
    return ChatResponse(**result)


@router.post("/chat/langchain", response_model=ChatResponse)
def chat_langchain(req: ChatRequest) -> ChatResponse:
    try:
        result = state.lc_rag_chain.invoke(req.question)
    except Exception:
        result = {"answer": _ERR_RAG, "retrieved_context": "", "used_rag": False}
    return ChatResponse(**result)


@router.post("/chat/langgraph", response_model=ChatResponse)
def chat_langgraph(req: ChatRequest) -> ChatResponse:
    try:
        result = state.lg_graph.invoke({
            "query": req.question,
            "documents": [],
            "score": 0.0,
            "answer": "",
            "used_rag": False,
            "retry_count": 0,
        })
        return ChatResponse(
            answer=result["answer"],
            retrieved_context=result["documents"][0] if result["used_rag"] and result["documents"] else "",
            used_rag=result["used_rag"],
        )
    except Exception:
        return ChatResponse(answer=_ERR_GRAPH, retrieved_context="", used_rag=False)


@router.post("/chat/claude/basic", response_model=ChatResponse)
def chat_claude_basic(req: ChatRequest) -> ChatResponse:
    answer = ask_claude(req.question)
    return ChatResponse(answer=answer, retrieved_context="", used_rag=False)


@router.post("/chat/claude/rag", response_model=ChatResponse)
def chat_claude_rag(req: ChatRequest) -> ChatResponse:
    try:
        result = state.claude_tfidf_chain(req.question)
    except Exception:
        result = {"answer": _ERR_RAG, "retrieved_context": "", "used_rag": False}
    return ChatResponse(**result)


@router.post("/chat/claude/langchain", response_model=ChatResponse)
def chat_claude_langchain(req: ChatRequest) -> ChatResponse:
    try:
        result = state.claude_lc_chain(req.question)
    except Exception:
        result = {"answer": _ERR_RAG, "retrieved_context": "", "used_rag": False}
    return ChatResponse(**result)


@router.post("/chat/claude/langgraph", response_model=ChatResponse)
def chat_claude_langgraph(req: ChatRequest) -> ChatResponse:
    try:
        result = state.claude_graph.invoke({
            "query": req.question,
            "documents": [],
            "score": 0.0,
            "answer": "",
            "used_rag": False,
            "retry_count": 0,
        })
        return ChatResponse(
            answer=result["answer"],
            retrieved_context=result["documents"][0] if result["used_rag"] and result["documents"] else "",
            used_rag=result["used_rag"],
        )
    except Exception:
        return ChatResponse(answer=_ERR_GRAPH, retrieved_context="", used_rag=False)
