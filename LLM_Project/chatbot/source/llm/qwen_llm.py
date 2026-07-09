"""Qwen3 모델 래퍼 — GGUF(llama-cpp)와 transformers(MPS) 통일 인터페이스."""

import re
from pathlib import Path

MODEL_DIR  = Path(__file__).resolve().parent.parent / "model" / "qwen"
BF16_DIR   = MODEL_DIR / "Qwen3-1.7B"          # transformers 원본 (비양자화)
Q4_PATH    = MODEL_DIR / "Qwen3-1.7B-Q4_K_M.gguf"

MAX_TOKENS       = 512
MAX_TOKENS_THINK = 1024   # thinking 모드는 CoT 토큰도 포함되므로 여유 확보
CONTEXT_SIZE     = 2048

_THINK_THRESHOLD = 30  # 이 글자 수 이상이고 물음표 포함이면 thinking 활성화


def _strip_think(text: str) -> str:
    """Qwen3 chain-of-thought <think>...</think> 블록 제거.

    max_tokens 초과로 </think> 없이 잘린 경우도 처리한다.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*",          "", text, flags=re.DOTALL)
    return text.strip()


# ── 공통 베이스 ────────────────────────────────────────────────────────────────

class QwenBase:
    def ask(self, question: str) -> str:
        raise NotImplementedError

    def ask_with_context(self, question: str, context: str) -> str:
        raise NotImplementedError


# ── GGUF 버전 (llama-cpp, Q4) ─────────────────────────────────────────────────

class QwenGGUF(QwenBase):
    def __init__(self, gguf_path: Path, n_gpu_layers: int = -1, verbose: bool = False):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=CONTEXT_SIZE,
            verbose=verbose,
        )

    @staticmethod
    def _needs_thinking(question: str) -> bool:
        """복잡한 사실형 질문이면 True — thinking 활성화 여부 자동 판단."""
        return len(question) >= _THINK_THRESHOLD and "?" in question

    def _chat(self, messages: list, think: bool) -> str:
        # /no_think or /think 마커를 마지막 user 메시지 끝에 삽입
        msgs = [m.copy() for m in messages]
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + (" /think" if think else " /no_think")
                break
        result = self.llm.create_chat_completion(
            messages=msgs,
            max_tokens=MAX_TOKENS_THINK if think else MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
        )
        text = _strip_think(result["choices"][0]["message"]["content"])
        # thinking이 토큰을 소진해 빈 답변이 나오면 no_think로 폴백
        if not text and think:
            return self._chat(messages, think=False)
        return text

    def ask(self, question: str) -> str:
        think = self._needs_thinking(question)
        sys_prompt = (
            "당신은 한국어·영어 질의응답 도우미입니다. 한국어와 영어로 답변하세요. 간결하게 답변하세요."
            if think else
            "당신은 한국어 질의응답 도우미입니다. 간결하게 답변하세요."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": question},
        ]
        try:
            return self._chat(messages, think=think)
        except Exception as e:
            return f"[오류] Qwen 응답 실패: {e}"

    def ask_with_context(self, question: str, context: str) -> str:
        # context가 주어지면 단순 추출 → thinking 불필요
        messages = [
            {"role": "system", "content": "당신은 한국어 질의응답 도우미입니다. 참고 문서를 바탕으로 간결하게 답변하세요."},
            {"role": "user",   "content": f"참고: {context}\n\n질문: {question}"},
        ]
        try:
            return self._chat(messages, think=False)
        except Exception as e:
            return f"[오류] Qwen 응답 실패: {e}"


# ── transformers 버전 (MPS, BF16 비양자화) ────────────────────────────────────

class QwenTransformers(QwenBase):
    def __init__(self, model_dir: Path = BF16_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

    def _generate(self, messages: list) -> str:
        import torch
        try:
            # tokenize=False로 문자열을 먼저 받고 별도 토크나이징
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[-1]

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[0][prompt_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            return f"[오류] Qwen 응답 실패: {e}"

    def ask(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "당신은 한국어 질의응답 도우미입니다. 간결하게 답변하세요."},
            {"role": "user",   "content": question},
        ]
        return self._generate(messages)

    def ask_with_context(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "당신은 한국어 질의응답 도우미입니다. 참고 문서를 바탕으로 간결하게 답변하세요."},
            {"role": "user",   "content": f"참고: {context}\n\n질문: {question}"},
        ]
        return self._generate(messages)
