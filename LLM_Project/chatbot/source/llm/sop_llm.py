"""LangChain 커스텀 컴포넌트.

SOP_GPT_LLM      : SopGptForCausalLM을 LangChain LLM 인터페이스로 래핑
make_span_extractor : SopGptForSpanExtraction을 LCEL RunnableLambda 주입용 함수로 반환
"""

import unicodedata
from typing import Any, Callable, Generator, List, Optional

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict

EOS_TOKEN = "<|endoftext|>"


class SOP_GPT_LLM(LLM):
    """SopGptForCausalLM 모델을 LangChain LLM 인터페이스로 래핑.

    stop_on="line"     → \\n으로 끝나는 토큰에서 멈춤  (QA 모드)
    stop_on="sentence" → .?!로 끝나는 토큰에서 멈춤  (이어쓰기 모드)
    stop_on="none"     → 멈춤 조건 없음
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model: Any
    tokenizer: Any
    stop_on: str = "line"
    temperature: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = None
    repetition_penalty: float = 1.3
    max_new_tokens: int = 60
    min_new_tokens: int = 0

    @property
    def _llm_type(self) -> str:
        return "sop_gpt"

    def _stop_tokens(self) -> set[int] | None:
        vocab = self.tokenizer.get_vocab()
        if self.stop_on == "line":
            ids = {i for t, i in vocab.items() if t.endswith("\n")}
        elif self.stop_on == "sentence":
            ids = {i for t, i in vocab.items() if t and t[-1] in ".?!"}
        else:
            ids = set()
        eos_id = vocab.get(EOS_TOKEN)
        if eos_id is not None:
            ids.add(eos_id)
        return ids or None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            stop_tokens = self._stop_tokens()
            ids = self.tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            device = next(self.hf_model.parameters()).device
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = self.hf_model.generate_raw(
                idx, self.max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                min_new_tokens=self.min_new_tokens,
            )[0].tolist()
            return self.tokenizer.decode(out[len(ids):]).strip()
        except torch.cuda.OutOfMemoryError:
            return "[오류] GPU 메모리가 부족합니다."
        except Exception as e:
            return f"[오류] 모델 추론 중 오류가 발생했습니다. ({type(e).__name__})"

    def stream_tokens(self, prompt: str) -> Generator[str, None, None]:
        """토큰 생성마다 현재까지 디코딩된 전체 텍스트를 yield하는 동기 제너레이터."""
        try:
            stop_tokens = self._stop_tokens()
            ids = self.tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            device = next(self.hf_model.parameters()).device
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            generated: list[int] = []
            for token_id in self.hf_model.generate_stream(
                idx, self.max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                min_new_tokens=self.min_new_tokens,
            ):
                generated.append(token_id)
                yield self.tokenizer.decode(generated).strip()
        except torch.cuda.OutOfMemoryError:
            yield "[오류] GPU 메모리가 부족합니다."
        except Exception as e:
            yield f"[오류] 모델 추론 중 오류가 발생했습니다. ({type(e).__name__})"


def make_span_extractor(span_model: Any, tokenizer: Any) -> Callable[[dict], str]:
    """SopGptForSpanExtraction을 {"question", "context"} → str 함수로 반환.

    chain.py 의 RunnableLambda 에 주입해 LCEL 체인 안에서 사용한다.
    """
    block_size = span_model.config.block_size

    def extract(inputs: dict) -> str:
        question = inputs["question"]
        context  = inputs["context"]
        prefix   = f"질문: {question}\n참고: "
        prompt   = prefix + context

        tokens, offsets, decomposed = tokenizer.tokenize_with_offsets(prompt)
        ids     = tokenizer.convert_tokens_to_ids(tokens)[:block_size]
        offsets = offsets[:block_size]

        nfd_prefix_len      = len(unicodedata.normalize("NFD", prefix))
        context_token_start = next(
            (i for i, (s, e) in enumerate(offsets) if e > nfd_prefix_len), 0
        )

        device = next(span_model.parameters()).device
        idx    = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = span_model(idx)

        start_logits = out.start_logits[0].float().clone()
        end_logits   = out.end_logits[0].float().clone()

        start_logits[:context_token_start] = float("-inf")
        end_logits[:context_token_start]   = float("-inf")

        start_idx = torch.argmax(start_logits).item()
        end_logits[:start_idx] = float("-inf")
        end_idx = torch.argmax(end_logits).item()

        char_start = offsets[start_idx][0]
        char_end   = offsets[end_idx][1]
        return unicodedata.normalize("NFC", decomposed[char_start:char_end])

    return extract
