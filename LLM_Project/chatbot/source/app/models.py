from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200


class GenerateResponse(BaseModel):
    text: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: str
    used_rag: bool
