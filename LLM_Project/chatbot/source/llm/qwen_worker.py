"""Qwen GGUF(llama.cpp)를 별도 프로세스에서 실행하는 프록시.

llama.cpp의 OpenMP 스레드가 PyTorch(SOP GPT)와 같은 프로세스에서 경쟁하지 않도록
spawn 컨텍스트로 자식 프로세스를 띄우고 큐로 요청/응답을 주고받는다.
QwenBase와 동일한 ask/ask_with_context 인터페이스라 호출부는 무수정.
"""

import multiprocessing as mp
import threading
from pathlib import Path

from llm.qwen_llm import QwenBase

_READY_TIMEOUT = 120  # 자식 프로세스 시작 대기 최대 시간(초)
_CALL_TIMEOUT  = 180  # 단일 추론 최대 대기 시간(초)


def _worker_main(gguf_path, n_gpu_layers, verbose, req_q, resp_q):
    from llm.qwen_llm import QwenGGUF
    qwen = QwenGGUF(Path(gguf_path), n_gpu_layers=n_gpu_layers, verbose=verbose)
    resp_q.put("ready")
    while True:
        item = req_q.get()
        if item is None:
            break
        method, args = item
        try:
            result = getattr(qwen, method)(*args)
        except Exception as e:
            result = f"[오류] Qwen 응답 실패: {e}"
        resp_q.put(result)


class QwenGGUFProcess(QwenBase):
    def __init__(self, gguf_path, n_gpu_layers=0, verbose=False):
        ctx = mp.get_context("spawn")
        self._req_q  = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._call_lock = threading.Lock()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(str(gguf_path), n_gpu_layers, verbose, self._req_q, self._resp_q),
            daemon=True,
        )
        self._proc.start()
        ready = self._resp_q.get(timeout=_READY_TIMEOUT)
        if ready != "ready":
            raise RuntimeError(f"QwenGGUFProcess 자식 프로세스 초기화 실패: {ready}")

    def _call(self, method, *args) -> str:
        with self._call_lock:
            self._req_q.put((method, args))
            return self._resp_q.get(timeout=_CALL_TIMEOUT)

    def ask(self, question: str) -> str:
        return self._call("ask", question)

    def ask_with_context(self, question: str, context: str) -> str:
        return self._call("ask_with_context", question, context)
