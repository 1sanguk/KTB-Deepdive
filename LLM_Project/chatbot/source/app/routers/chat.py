import traceback
import uuid
from fastapi import APIRouter, Query

import state
from auth import login_or_register
from llm.claude_llm import ask_claude
from models import (
    AuthRequest, AuthResponse, ChatRequest, ChatResponse,
    GenerateRequest, GenerateResponse,
    SessionCreateRequest, SessionRenameRequest,
)
from sessions import list_sessions, create_session, rename_session

router = APIRouter()

_ERR_SOP  = "[오류] 모델 추론 중 오류가 발생했습니다."
_ERR_RAG  = "[오류] 검색 중 오류가 발생했습니다."
_ERR_GRAPH = "[오류] 파이프라인 실행 중 오류가 발생했습니다."


def _tid(thread_id: str) -> str:
    return thread_id.strip().lower()


@router.post("/auth/login", response_model=AuthResponse)
def auth_login(req: AuthRequest) -> AuthResponse:
    ok, msg = login_or_register(req.user_id, req.password)
    return AuthResponse(ok=ok, msg=msg)


# ── 세션 관리 ──────────────────────────────────────────────────────────────────

@router.get("/sessions")
def get_sessions(user_id: str = Query(...)):
    return {"sessions": list_sessions(user_id.strip().lower())}


@router.post("/sessions")
def post_session(req: SessionCreateRequest):
    return create_session(req.user_id, req.name)


@router.patch("/sessions/{session_id}")
def patch_session(session_id: str, req: SessionRenameRequest):
    ok = rename_session(req.user_id, session_id, req.name)
    return {"ok": ok}


@router.get("/chat/compare/history")
def get_compare_history(thread_id: str = Query(...)):
    return {"messages": state.load_history(thread_id.strip().lower())}


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
        tid = req.thread_id or str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid}}
        result = state.lg_graph.invoke({
            "query": req.question,
            "documents": [],
            "score": 0.0,
            "answer": "",
            "used_rag": False,
            "retry_count": 0,
            "messages": [],
        }, config=cfg)
        return ChatResponse(
            answer=result["answer"],
            retrieved_context=result["documents"][0] if result["used_rag"] and result["documents"] else "",
            used_rag=result["used_rag"],
        )
    except Exception:
        print(traceback.format_exc())
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
        tid = (req.thread_id + ":c") if req.thread_id else str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid}}
        result = state.claude_graph.invoke({
            "query": req.question,
            "documents": [],
            "score": 0.0,
            "answer": "",
            "used_rag": False,
            "retry_count": 0,
            "messages": [],
        }, config=cfg)
        return ChatResponse(
            answer=result["answer"],
            retrieved_context=result["documents"][0] if result["used_rag"] and result["documents"] else "",
            used_rag=result["used_rag"],
        )
    except Exception:
        print(traceback.format_exc())
        return ChatResponse(answer=_ERR_GRAPH, retrieved_context="", used_rag=False)


@router.get("/chat/basic/history")
def get_basic_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid)}

@router.get("/chat/claude/basic/history")
def get_claude_basic_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid + ":c")}

@router.get("/chat/rag/history")
def get_rag_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid)}

@router.get("/chat/claude/rag/history")
def get_claude_rag_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid + ":c")}

@router.get("/chat/langchain/history")
def get_langchain_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid)}

@router.get("/chat/claude/langchain/history")
def get_claude_langchain_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid + ":c")}

@router.get("/chat/langgraph/history")
def get_langgraph_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    # MemorySaver에 당세션 데이터 있으면 우선 사용, 없으면 JSON 파일에서 로드
    try:
        snap = state.lg_graph.get_state({"configurable": {"thread_id": tid}})
        if snap and snap.values and snap.values.get("messages"):
            return {"messages": snap.values["messages"]}
    except Exception:
        pass
    return {"messages": state.load_history(tid)}


@router.get("/chat/claude/langgraph/history")
def get_claude_langgraph_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    claude_tid = tid + ":c"
    try:
        snap = state.claude_graph.get_state({"configurable": {"thread_id": claude_tid}})
        if snap and snap.values and snap.values.get("messages"):
            return {"messages": snap.values["messages"]}
    except Exception:
        pass
    return {"messages": state.load_history(claude_tid)}


@router.get("/chat/auto/history")
def get_auto_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid)}


@router.get("/chat/claude/auto/history")
def get_claude_auto_history(thread_id: str = Query(...)):
    tid = _tid(thread_id)
    return {"messages": state.load_history(tid + ":c")}
