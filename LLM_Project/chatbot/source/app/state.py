"""서버 시작 시 한 번만 로드되는 모델·체인·검색기 전역 상태."""

import os
import sys
import threading
# import importlib.util  # vLLM 감지에 사용 (vLLM 활성화 시 주석 해제)
import torch
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent
_HF_PKG = _SOURCE / "model" / "sop_gpt_hf"
# sop_model/model.py가 'model' 모듈로 인식되는 충돌 방지 — 패키지 경로 직접 주입
sys.path.insert(0, str(_HF_PKG))

from modeling_sop_gpt import SopGptForCausalLM, SopGptForSpanExtraction  # noqa: E402
from tokenization_sop_gpt import SopGptTokenizer                           # noqa: E402
from llm.sop_llm import SOP_GPT_LLM, make_span_extractor
from lc.chain import build_basic_chain, build_rag_chain
from lc.retriever import build_hybrid_retriever
from llm.claude_llm import build_claude_rag_chain
from llm.qwen_llm import QwenTransformers, QwenGGUF, BF16_DIR, Q4_PATH
# from llm.vllm_llm import VLLMQwen  # vLLM 활성화 시 주석 해제
from rag.rag import build_tfidf_retriever
from lg.graph import build_graph, build_claude_graph, build_qwen_graph, build_claude_agent_graph
from history import load_history, save_history, append_history
from langgraph.checkpoint.memory import MemorySaver

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

HF_GEN_ID   = "1sangukn/sop-gpt"
HF_QA_ID    = "1sangukn/sop-gpt-qa"
HF_SPAN_ID  = "1sangukn/sop-gpt-span"

RAG_SIM_THRESHOLD   = 0.515
TFIDF_SIM_THRESHOLD = 0.25
GRAPH_SOP_THRESHOLD    = [0.35, 0.25, 0.2]
GRAPH_CLAUDE_THRESHOLD = [0.515, 0.375, 0.350]

now_count   = 0
total_count = 11  # gen_model은 lazy load이므로 카운트에서 제외

# ── 토크나이저 (gen·qa·span 공통) ──────────────────────────────────────────────
print(f"[{now_count}/{total_count}] 토크나이저 로딩 중...")
tokenizer = SopGptTokenizer.from_pretrained(HF_GEN_ID)
now_count += 1
print(f"[{now_count}/{total_count}] 토크나이저 로딩 완료.")

# float16으로 직접 로드 (캐시 없으면 HF Hub에서 자동 다운로드)
_hf_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# ── gen_model: /generate 전용 — lazy load (서버 시작 시 ~180MB RAM 절약) ─────────
_gen_lock   = threading.Lock()
_gen_loaded = False
gen_model   = None
gen_llm     = None

def load_gen_model():
    """최초 /generate 요청 시 1회 로드. 이후 호출은 즉시 반환."""
    global _gen_loaded, gen_model, gen_llm
    if _gen_loaded:
        return
    with _gen_lock:
        if _gen_loaded:
            return
        print("[lazy] 이어쓰기 모델(gen) 로딩 중...")
        gen_model = SopGptForCausalLM.from_pretrained(HF_GEN_ID, **_hf_kwargs).eval().to(device)
        gen_llm = SOP_GPT_LLM(
            hf_model=gen_model, tokenizer=tokenizer,
            stop_on="sentence", temperature=0.7, top_k=None, top_p=0.9,
            repetition_penalty=1.3, max_new_tokens=200,
        )
        _gen_loaded = True
        print("[lazy] 이어쓰기 모델(gen) 로딩 완료.")

# ── HF Hub 모델 로드 ───────────────────────────────────────────────────────────
print(f"[{now_count}/{total_count}] QA 모델 로딩 중...")
qa_model = SopGptForCausalLM.from_pretrained(HF_QA_ID, **_hf_kwargs).eval().to(device)
now_count += 1
print(f"[{now_count}/{total_count}] QA 모델 로딩 완료.")

print(f"[{now_count}/{total_count}] Span 추출 모델 로딩 중...")
span_model = SopGptForSpanExtraction.from_pretrained(HF_SPAN_ID, **_hf_kwargs).eval().to(device)
now_count += 1
print(f"[{now_count}/{total_count}] Span 추출 모델 로딩 완료.")

# ── 검색기 ─────────────────────────────────────────────────────────────────────
print(f"[{now_count}/{total_count}] TF-IDF 검색기 로딩 중...")
tfidf_retriever = build_tfidf_retriever()
now_count += 1
print(f"[{now_count}/{total_count}] TF-IDF 검색기 로딩 완료.")

print(f"[{now_count}/{total_count}] 하이브리드 검색기(BM25+FAISS) 로딩 중...")
lc_retriever = build_hybrid_retriever()
now_count += 1
print(f"[{now_count}/{total_count}] 하이브리드 검색기(BM25+FAISS) 로딩 완료.")

# ── LangChain LLM ──────────────────────────────────────────────────────────────
qa_llm = SOP_GPT_LLM(
    hf_model=qa_model, tokenizer=tokenizer,
    stop_on="sentence", temperature=0.7, top_k=40, top_p=0.9,
    repetition_penalty=1.3, max_new_tokens=250, min_new_tokens=20,
)
span_extractor_fn = make_span_extractor(span_model, tokenizer)

# ── LCEL 체인 ──────────────────────────────────────────────────────────────────
basic_chain     = build_basic_chain(qa_llm)
tfidf_rag_chain = build_rag_chain(tfidf_retriever, qa_llm, span_extractor_fn, TFIDF_SIM_THRESHOLD)
lc_rag_chain    = build_rag_chain(lc_retriever,    qa_llm, span_extractor_fn, RAG_SIM_THRESHOLD)

# ── Claude 체인 ────────────────────────────────────────────────────────────────
claude_tfidf_chain = build_claude_rag_chain(tfidf_retriever, TFIDF_SIM_THRESHOLD)
claude_lc_chain    = build_claude_rag_chain(lc_retriever,    RAG_SIM_THRESHOLD)

# ── vLLM / Ollama 외부 서버 모드 (GPU 인스턴스 등에서 사용) ──────────────────────
# _VLLM_URL_ENV   = os.environ.get("VLLM_BASE_URL", "")
# _VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None
#
# # VLLM_BASE_URL 명시 → Ollama/vLLM 모두 커버
# # vllm 패키지 설치됨 → 기본 vLLM 포트(8001) 시도
# # 둘 다 없으면 → 로컬 모델
# if _VLLM_URL_ENV:
#     _VLLM_URL = _VLLM_URL_ENV
# elif _VLLM_INSTALLED:
#     _VLLM_URL = "http://localhost:8001/v1"
# else:
#     _VLLM_URL = ""
#
# if _VLLM_URL:
#     # ── 외부 서버 모드 (vLLM 또는 Ollama) ────────────────────────────────────
#     print(f"[{now_count}/{total_count}] 외부 LLM 서버 연결 중... ({_VLLM_URL})")
#     _vllm = VLLMQwen()
#     if _vllm.health_check():
#         qwen_llm       = _vllm
#         qwen_quant_llm = _vllm   # 단일 서버를 두 슬롯 모두에 매핑
#         now_count += 1
#         print(f"[{now_count}/{total_count}] 외부 LLM 서버 연결 완료 — qwen·qwen-q 슬롯 사용.")
#     else:
#         print(f"[경고] 서버({_VLLM_URL})에 연결할 수 없음 — 로컬 모델로 폴백")
#         qwen_llm       = None
#         qwen_quant_llm = None
# else:

# ── 로컬 모델 로딩 (t3.medium 등 vLLM 미사용 환경) ─────────────────────────────
if BF16_DIR.exists():
    print(f"[{now_count}/{total_count}] Qwen BF16 (비양자화) 로딩 중...")
    qwen_llm = QwenTransformers(BF16_DIR)
    now_count += 1
    print(f"[{now_count}/{total_count}] Qwen BF16 (비양자화) 로딩 완료.")
else:
    print(f"[skip] Qwen BF16 모델 없음 — qwen 엔드포인트 비활성화")
    qwen_llm = None

if Q4_PATH.exists():
    print(f"[{now_count}/{total_count}] Qwen Q4_K_M (양자화) 로딩 중...")
    qwen_quant_llm = QwenGGUF(Q4_PATH, verbose=False)
    now_count += 1
    print(f"[{now_count}/{total_count}] Qwen Q4_K_M (양자화) 로딩 완료.")
else:
    print(f"[skip] Qwen Q4 모델 없음 — qwen-q 엔드포인트 비활성화")
    qwen_quant_llm = None

print(f"[{now_count}/{total_count}] LangGraph 파이프라인 빌드 중...")
lg_graph = build_graph(lc_retriever, qa_llm, span_extractor_fn, GRAPH_SOP_THRESHOLD,
                       checkpointer=MemorySaver())
now_count += 1
print(f"[{now_count}/{total_count}] LangGraph 파이프라인 빌드 완료.")

print(f"[{now_count}/{total_count}] Claude LangGraph 파이프라인 빌드 중...")
# JSON 정준 히스토리에서 직접 시드하므로 MemorySaver 불필요 (있으면 중복 누적)
claude_graph = build_claude_graph(lc_retriever, GRAPH_CLAUDE_THRESHOLD)
now_count += 1
print(f"[{now_count}/{total_count}] Claude LangGraph 파이프라인 빌드 완료.")

print(f"[{now_count}/{total_count}] Claude Agent Graph 빌드 중...")
claude_agent_graph = build_claude_agent_graph(lc_retriever)
now_count += 1
print(f"[{now_count}/{total_count}] Claude Agent Graph 빌드 완료.")

print(f"[{now_count}/{total_count}] Qwen LangGraph 파이프라인 빌드 중...")
qwen_graph = build_qwen_graph(lc_retriever, qwen_llm, GRAPH_CLAUDE_THRESHOLD) if qwen_llm else None
qwen_quant_graph = build_qwen_graph(lc_retriever, qwen_quant_llm, GRAPH_CLAUDE_THRESHOLD) if qwen_quant_llm else None
now_count += 1
print(f"[{now_count}/{total_count}] Qwen LangGraph 파이프라인 빌드 완료.")

LANGGRAPH_GRAPHS = {k: v for k, v in {
    "sop":    lg_graph,
    "claude": claude_graph,
    "qwen":   qwen_graph,
    "qwen-q": qwen_quant_graph,
}.items() if v is not None}

THREAD_SUFFIXES = {
    "sop":    "",
    "claude": ":c",
    "qwen":   ":bf16",
    "qwen-q": ":q4",
}

print("=" * 40)
print("서버 준비 완료. 요청을 받을 수 있습니다.")
print("=" * 40)
