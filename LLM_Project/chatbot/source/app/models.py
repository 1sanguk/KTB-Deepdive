from pydantic import BaseModel, field_validator


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200


class GenerateResponse(BaseModel):
    text: str


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    model: str = "sop"  # langgraph 엔드포인트용: "sop" | "claude" | "qwen" | "qwen-q"

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else None


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: str
    used_rag: bool


class AuthRequest(BaseModel):
    user_id: str
    password: str

    @field_validator("user_id")
    @classmethod
    def normalize_user_id(cls, v: str) -> str:
        return v.strip().lower()


class AuthResponse(BaseModel):
    ok: bool
    msg: str  # "registered" | "logged_in" | "wrong_password"
