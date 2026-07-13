"""서버 시작 시 한 번만 로드되는 모델·체인·검색기 전역 상태."""

import torch
from pathlib import Path

from model import device, SOP_GPT, SOP_GPT_Span
from bpe import build_vocab, base_alphabet, load_bpe
from llm.sop_llm import SOP_GPT_LLM, make_span_extractor
from lc.chain import build_basic_chain, build_rag_chain
from lc.retriever import build_hybrid_retriever
from llm.claude_llm import build_claude_rag_chain
from llm.qwen_llm import QwenTransformers, QwenGGUF, BF16_DIR, Q4_PATH
from rag.rag import build_tfidf_retriever
from lg.graph import build_graph, build_claude_graph, build_qwen_graph, build_claude_agent_graph
from history import load_history, save_history, append_history
from langgraph.checkpoint.memory import MemorySaver

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "sop_model"

BPE_PATH  = MODEL_DIR / "bpe_vocab.json"
GEN_CKPT  = MODEL_DIR / "SOP_GPT.pt"
QA_CKPT   = MODEL_DIR / "SOP_GPT_qa.pt"
SPAN_CKPT = MODEL_DIR / "SOP_GPT_span.pt"

RAG_SIM_THRESHOLD   = 0.515   # BM25+FAISS 하이브리드 임계값
TFIDF_SIM_THRESHOLD = 0.25    # TF-IDF 단독 임계값
GRAPH_SOP_THRESHOLD    = [0.35, 0.25, 0.2]   # SOP 모델: RAG 의존도 높으므로 낮게
GRAPH_CLAUDE_THRESHOLD = [0.515, 0.375, 0.350]  # Claude: 자체 생성 능력 있으므로 높게 / retry는 F1 최적 구간(0.325~0.375) 기준

now_count   = 0
total_count = 12

# ── BPE 토크나이저 ─────────────────────────────────────────────────────────────
print(f"[{now_count}/{total_count}] BPE 토크나이저 로딩 중...")
vocab, merges = load_bpe(BPE_PATH)
stoi, itos    = build_vocab(vocab)
vocab_size    = len(vocab)
base_set      = base_alphabet(vocab)
now_count += 1
print(f"[{now_count}/{total_count}] BPE 토크나이저 로딩 완료.")

# ── PyTorch 모델 ───────────────────────────────────────────────────────────────
print(f"[{now_count}/{total_count}] 이어쓰기 모델(gen) 로딩 중...")
gen_model = SOP_GPT(vocab_size).to(device)
gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
gen_model = gen_model.half()
gen_model.eval()
now_count += 1
print(f"[{now_count}/{total_count}] 이어쓰기 모델(gen) 로딩 완료.")

print(f"[{now_count}/{total_count}] QA 모델 로딩 중...")
qa_model = SOP_GPT(vocab_size).to(device)
qa_model.load_state_dict(torch.load(QA_CKPT, map_location=device))
qa_model = qa_model.half()
qa_model.eval()
now_count += 1
print(f"[{now_count}/{total_count}] QA 모델 로딩 완료.")

print(f"[{now_count}/{total_count}] Span 추출 모델 로딩 중...")
span_model = SOP_GPT_Span(vocab_size).to(device)
span_model.load_state_dict(torch.load(SPAN_CKPT, map_location=device))
span_model = span_model.half()
span_model.eval()
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
gen_llm = SOP_GPT_LLM(
    torch_model=gen_model, stoi=stoi, itos=itos, merges=merges, base_set=base_set,
    stop_on="sentence", temperature=0.7, top_k=None, top_p=0.9,
    repetition_penalty=1.3, max_new_tokens=200,
)
qa_llm = SOP_GPT_LLM(
    torch_model=qa_model, stoi=stoi, itos=itos, merges=merges, base_set=base_set,
    stop_on="sentence", temperature=0.7, top_k=40, top_p=0.9,
    repetition_penalty=1.3, max_new_tokens=250, min_new_tokens=20,
)
span_extractor_fn = make_span_extractor(span_model, stoi, merges, base_set)

# ── LCEL 체인 ──────────────────────────────────────────────────────────────────
basic_chain     = build_basic_chain(qa_llm)
tfidf_rag_chain = build_rag_chain(tfidf_retriever, qa_llm, span_extractor_fn, TFIDF_SIM_THRESHOLD)
lc_rag_chain    = build_rag_chain(lc_retriever,    qa_llm, span_extractor_fn, RAG_SIM_THRESHOLD)

# ── Claude 체인 ────────────────────────────────────────────────────────────────
claude_tfidf_chain = build_claude_rag_chain(tfidf_retriever, TFIDF_SIM_THRESHOLD)
claude_lc_chain    = build_claude_rag_chain(lc_retriever,    RAG_SIM_THRESHOLD)

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
claude_graph = build_claude_graph(lc_retriever, GRAPH_CLAUDE_THRESHOLD,
                                  checkpointer=MemorySaver())
now_count += 1
print(f"[{now_count}/{total_count}] Claude LangGraph 파이프라인 빌드 완료.")

print(f"[{now_count}/{total_count}] Claude Agent Graph 빌드 중...")
claude_agent_graph = build_claude_agent_graph(lc_retriever)
now_count += 1
print(f"[{now_count}/{total_count}] Claude Agent Graph 빌드 완료.")

print(f"[{now_count}/{total_count}] Qwen LangGraph 파이프라인 빌드 중...")
qwen_graph = build_qwen_graph(lc_retriever, qwen_llm, GRAPH_CLAUDE_THRESHOLD,
                              checkpointer=MemorySaver()) if qwen_llm else None
qwen_quant_graph = build_qwen_graph(lc_retriever, qwen_quant_llm, GRAPH_CLAUDE_THRESHOLD,
                                    checkpointer=MemorySaver()) if qwen_quant_llm else None
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
