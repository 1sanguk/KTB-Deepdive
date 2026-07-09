"""LangChain 커스텀 컴포넌트.

SOP_GPT_LLM  : SOP_GPT QA/Gen 모델을 LangChain LLM 인터페이스로 래핑
make_span_extractor : SOP_GPT_Span 추출형 QA를 LCEL RunnableLambda 주입용 함수로 반환
"""

import sys
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict

SOURCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_DIR / "model"))

from bpe import tokenize, decode, EOS_TOKEN
from model import device
from chat import extract_answer as _extract_answer


class SOP_GPT_LLM(LLM):
    """SOP_GPT 모델을 LangChain LLM 인터페이스로 래핑.

    stop_on="line"     → \\n으로 끝나는 토큰에서 멈춤  (QA 모드, Stage 2)
    stop_on="sentence" → .?!로 끝나는 토큰에서 멈춤  (이어쓰기 모드, Stage 1)
    stop_on="none"     → 멈춤 조건 없음
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    torch_model: Any
    stoi: Any
    itos: Any
    merges: Any
    base_set: Any
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
        """stop_on 설정에 따른 stop token id 집합을 반환. EOS 토큰은 항상 포함."""
        if self.stop_on == "line":
            ids = {i for t, i in self.stoi.items() if t.endswith("\n")}
        elif self.stop_on == "sentence":
            ids = {i for t, i in self.stoi.items() if t and t[-1] in ".?!"}
        else:
            ids = set()
        eos_id = self.stoi.get(EOS_TOKEN)
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
            ids = [self.stoi[t] for t in tokenize(prompt, self.merges, self.base_set) if t in self.stoi]
            if not ids:
                ids = [0]  # 입력이 vocab에 없을 때 START_ID 로 대체
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = self.torch_model.generate(
                idx, self.max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                min_new_tokens=self.min_new_tokens,
            )[0].tolist()
            return decode(out[len(ids):], self.itos).strip()
        except torch.cuda.OutOfMemoryError:
            return "[오류] GPU 메모리가 부족합니다."
        except Exception as e:
            return f"[오류] 모델 추론 중 오류가 발생했습니다. ({type(e).__name__})"

    def stream_tokens(self, prompt: str) -> Generator[str, None, None]:
        """토큰 생성마다 현재까지 디코딩된 전체 텍스트를 yield하는 동기 제너레이터."""
        try:
            stop_tokens = self._stop_tokens()
            ids = [self.stoi[t] for t in tokenize(prompt, self.merges, self.base_set) if t in self.stoi]
            if not ids:
                ids = [0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            generated: list[int] = []
            for token_id in self.torch_model.generate_stream(
                idx, self.max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                min_new_tokens=self.min_new_tokens,
            ):
                generated.append(token_id)
                yield decode(generated, self.itos).strip()
        except torch.cuda.OutOfMemoryError:
            yield "[오류] GPU 메모리가 부족합니다."
        except Exception as e:
            yield f"[오류] 모델 추론 중 오류가 발생했습니다. ({type(e).__name__})"


def make_span_extractor(span_model: Any, stoi: dict[str, int], merges: Any, base_set: set[str]) -> Callable[[dict], str]:
    """SOP_GPT_Span 추출형 QA를 {"question", "context"} → str 함수로 반환.

    chain.py 의 RunnableLambda 에 주입해 LCEL 체인 안에서 사용한다.
    """
    def extract(inputs: dict) -> str:
        return _extract_answer(
            span_model, stoi, merges, base_set,
            inputs["question"], inputs["context"],
        )
    return extract
