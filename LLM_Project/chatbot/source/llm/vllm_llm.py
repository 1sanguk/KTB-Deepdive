"""vLLM OpenAI-compatible API 클라이언트.

vLLM 서버를 별도로 띄운 뒤 VLLM_BASE_URL 환경변수를 설정하면
QwenBase와 동일한 인터페이스로 투명하게 교체된다.

서버 실행 예시 (GPU):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-1.7B \
        --port 8001 \
        --max-model-len 2048

서버 실행 예시 (CPU-only, GPU 없는 환경):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-1.7B \
        --device cpu \
        --dtype float32 \
        --port 8001 \
        --max-model-len 2048

로컬 모델 경로 사용 시 --model 에 절대 경로를 지정한다.
"""

import os
import re

import httpx

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen3-1.7B")

MAX_TOKENS    = 512
_TIMEOUT      = 120.0  # 초


def _strip_think(text: str) -> str:
    """Qwen3 chain-of-thought <think>…</think> 블록 제거."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*",          "", text, flags=re.DOTALL)
    return text.strip()


class VLLMQwen:
    """vLLM 서버에 chat/completions 요청을 보내는 동기 클라이언트.

    QwenBase(ask / ask_with_context)와 동일한 인터페이스를 유지하므로
    state.py에서 QwenTransformers / QwenGGUF와 drop-in 교체가 가능하다.
    """

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model:    str = VLLM_MODEL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self._client  = httpx.Client(timeout=_TIMEOUT)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "max_tokens":  MAX_TOKENS,
                    "temperature": temperature,
                    "top_p":       0.9,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            return _strip_think(content)
        except httpx.HTTPStatusError as e:
            return f"[오류] vLLM HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"[오류] vLLM 응답 실패: {type(e).__name__}: {e}"

    # ── 공개 인터페이스 (QwenBase 호환) ───────────────────────────────────────

    def ask(self, question: str) -> str:
        return self._chat([
            {"role": "system", "content": "당신은 한국어 질의응답 도우미입니다. 간결하게 답변하세요."},
            {"role": "user",   "content": question},
        ])

    def ask_with_context(self, question: str, context: str) -> str:
        return self._chat([
            {"role": "system", "content": "당신은 한국어 질의응답 도우미입니다. 참고 문서를 바탕으로 간결하게 답변하세요."},
            {"role": "user",   "content": f"참고: {context}\n\n질문: {question}"},
        ])

    def close(self) -> None:
        self._client.close()

    # ── 연결 테스트 ────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """vLLM 서버가 응답하는지 확인. state.py 기동 시 사용."""
        try:
            resp = self._client.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
