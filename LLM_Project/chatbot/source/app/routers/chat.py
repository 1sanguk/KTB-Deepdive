from fastapi import APIRouter

import state
from lc.claude_llm import ask_claude
from models import ChatRequest, ChatResponse, GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    text = state.gen_llm.invoke(req.prompt)
    return GenerateResponse(text=text)


@router.post("/chat/basic", response_model=ChatResponse)
def chat_basic(req: ChatRequest):
    answer = state.basic_chain.invoke(req.question)
    return ChatResponse(answer=answer, retrieved_context="", used_rag=False)


@router.post("/chat/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest):
    result = state.tfidf_rag_chain.invoke(req.question)
    return ChatResponse(**result)


@router.post("/chat/langchain", response_model=ChatResponse)
def chat_langchain(req: ChatRequest):
    result = state.lc_rag_chain.invoke(req.question)
    return ChatResponse(**result)


@router.post("/chat/claude/basic", response_model=ChatResponse)
def chat_claude_basic(req: ChatRequest):
    answer = ask_claude(req.question)
    return ChatResponse(answer=answer, retrieved_context="", used_rag=False)


@router.post("/chat/claude/rag", response_model=ChatResponse)
def chat_claude_rag(req: ChatRequest):
    result = state.claude_tfidf_chain(req.question)
    return ChatResponse(**result)


@router.post("/chat/claude/langchain", response_model=ChatResponse)
def chat_claude_langchain(req: ChatRequest):
    result = state.claude_lc_chain(req.question)
    return ChatResponse(**result)
