from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import state
from lc.claude_llm import stream_claude
from models import ChatRequest
from streaming import _sse, sop_stream, sop_rag_stream, sop_lg_stream, claude_rag_stream

router = APIRouter()


@router.post("/chat/basic/stream")
async def chat_basic_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        sop_stream(state.qa_llm, f"질문: {req.question}\n답변: "),
        media_type="text/event-stream",
    )


@router.post("/chat/rag/stream")
async def chat_rag_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        sop_rag_stream(
            state.tfidf_retriever, state.qa_llm,
            state.span_extractor_fn, state.TFIDF_SIM_THRESHOLD,
            req.question,
        ),
        media_type="text/event-stream",
    )


@router.post("/chat/langchain/stream")
async def chat_langchain_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        sop_rag_stream(
            state.lc_retriever, state.qa_llm,
            state.span_extractor_fn, state.RAG_SIM_THRESHOLD,
            req.question,
        ),
        media_type="text/event-stream",
    )


@router.post("/chat/langgraph/stream")
async def chat_langgraph_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        sop_lg_stream(state.lg_graph, req.question),
        media_type="text/event-stream",
    )


@router.post("/chat/claude/langgraph/stream")
async def chat_claude_langgraph_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        sop_lg_stream(state.claude_graph, req.question),
        media_type="text/event-stream",
    )


@router.post("/chat/claude/basic/stream")
async def chat_claude_basic_stream(req: ChatRequest) -> StreamingResponse:
    async def gen():
        async for text in stream_claude(req.question):
            yield _sse({"type": "text", "text": text})
        yield _sse({"type": "done"})
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/chat/claude/rag/stream")
async def chat_claude_rag_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        claude_rag_stream(state.tfidf_retriever, state.TFIDF_SIM_THRESHOLD, req.question),
        media_type="text/event-stream",
    )


@router.post("/chat/claude/langchain/stream")
async def chat_claude_langchain_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        claude_rag_stream(state.lc_retriever, state.RAG_SIM_THRESHOLD, req.question),
        media_type="text/event-stream",
    )
