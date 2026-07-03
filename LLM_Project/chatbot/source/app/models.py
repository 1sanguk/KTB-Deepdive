from pydantic import BaseModel, field_validator


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200


class GenerateResponse(BaseModel):
    text: str


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else None


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: str
    used_rag: bool
