"""서버 시작 시 한 번만 로드되는 모델·체인·검색기 전역 상태."""

import torch
from pathlib import Path

from model import device, SOP_GPT, SOP_GPT_Span
from bpe import build_vocab, base_alphabet, load_bpe
from lc.llm import SOP_GPT_LLM, make_span_extractor
from lc.chain import build_basic_chain, build_rag_chain
from lc.retriever import build_hybrid_retriever
from lc.claude_llm import build_claude_rag_chain
from rag.rag import build_tfidf_retriever
from lg.graph import build_graph, build_claude_graph
from langgraph.checkpoint.memory import MemorySaver
import json

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

BPE_PATH  = MODEL_DIR / "bpe_vocab.json"
GEN_CKPT  = MODEL_DIR / "SOP_GPT.pt"
QA_CKPT   = MODEL_DIR / "SOP_GPT_qa.pt"
SPAN_CKPT = MODEL_DIR / "SOP_GPT_span.pt"

RAG_SIM_THRESHOLD   = 0.515   # BM25+FAISS 하이브리드 임계값
TFIDF_SIM_THRESHOLD = 0.25    # TF-IDF 단독 임계값
GRAPH_SOP_THRESHOLD    = [0.35, 0.25, 0.2]   # SOP 모델: RAG 의존도 높으므로 낮게
GRAPH_CLAUDE_THRESHOLD = [0.515, 0.4, 0.3]  # Claude: 자체 생성 능력 있으므로 높게

now_count   = 0
total_count = 7

# ── BPE 토크나이저 ─────────────────────────────────────────────────────────────
now_count += 1
print(f"[{now_count}/{total_count}] BPE 토크나이저 로딩 중...")
vocab, merges = load_bpe(BPE_PATH)
stoi, itos    = build_vocab(vocab)
vocab_size    = len(vocab)
base_set      = base_alphabet(vocab)
print(f"[{now_count}/{total_count}] BPE 토크나이저 로딩 완료.")

# ── PyTorch 모델 ───────────────────────────────────────────────────────────────
now_count += 1
print(f"[{now_count}/{total_count}] 이어쓰기 모델(gen) 로딩 중...")
gen_model = SOP_GPT(vocab_size).to(device)
gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
gen_model.eval()
print(f"[{now_count}/{total_count}] 이어쓰기 모델(gen) 로딩 완료.")

now_count += 1
print(f"[{now_count}/{total_count}] QA 모델 로딩 중...")
qa_model = SOP_GPT(vocab_size).to(device)
qa_model.load_state_dict(torch.load(QA_CKPT, map_location=device))
qa_model.eval()
print(f"[{now_count}/{total_count}] QA 모델 로딩 완료.")

now_count += 1
print(f"[{now_count}/{total_count}] Span 추출 모델 로딩 중...")
span_model = SOP_GPT_Span(vocab_size).to(device)
span_model.load_state_dict(torch.load(SPAN_CKPT, map_location=device))
span_model.eval()
print(f"[{now_count}/{total_count}] Span 추출 모델 로딩 완료.")

# ── 검색기 ─────────────────────────────────────────────────────────────────────
now_count += 1
print(f"[{now_count}/{total_count}] TF-IDF 검색기 로딩 중...")
tfidf_retriever = build_tfidf_retriever()
print(f"[{now_count}/{total_count}] TF-IDF 검색기 로딩 완료.")

now_count += 1
print(f"[{now_count}/{total_count}] 하이브리드 검색기(BM25+FAISS) 로딩 중...")
lc_retriever = build_hybrid_retriever()
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
    repetition_penalty=1.3, max_new_tokens=250,
)
span_extractor_fn = make_span_extractor(span_model, stoi, merges, base_set)

# ── LCEL 체인 ──────────────────────────────────────────────────────────────────
basic_chain     = build_basic_chain(qa_llm)
tfidf_rag_chain = build_rag_chain(tfidf_retriever, qa_llm, span_extractor_fn, TFIDF_SIM_THRESHOLD)
lc_rag_chain    = build_rag_chain(lc_retriever,    qa_llm, span_extractor_fn, RAG_SIM_THRESHOLD)

# ── Claude 체인 ────────────────────────────────────────────────────────────────
claude_tfidf_chain = build_claude_rag_chain(tfidf_retriever, TFIDF_SIM_THRESHOLD)
claude_lc_chain    = build_claude_rag_chain(lc_retriever,    RAG_SIM_THRESHOLD)

# ── LangGraph 파이프라인 ────────────────────────────────────────────────────────
total_count = 8

_data_dir = Path(__file__).resolve().parent.parent / "data"
_history_dir = _data_dir / "history"
_history_dir.mkdir(parents=True, exist_ok=True)


def load_history(thread_id: str) -> list:
    """저장된 JSON 히스토리 반환. 없으면 빈 리스트."""
    path = _history_dir / f"{thread_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(thread_id: str, messages: list) -> None:
    """LangGraph용: MemorySaver 전체 메시지 리스트를 저장. 세션 내 누적이 더 길면 덮어쓰기."""
    _history_dir.mkdir(parents=True, exist_ok=True)
    path = _history_dir / f"{thread_id}.json"
    existing = load_history(thread_id)
    to_save = messages if len(messages) >= len(existing) else existing + messages
    path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(thread_id: str, new_messages: list) -> None:
    """Basic/RAG/LangChain용: 기존 히스토리에 새 메시지 이어붙이기."""
    _history_dir.mkdir(parents=True, exist_ok=True)
    existing = load_history(thread_id)
    path = _history_dir / f"{thread_id}.json"
    path.write_text(json.dumps(existing + new_messages, ensure_ascii=False, indent=2), encoding="utf-8")


now_count += 1
print(f"[{now_count}/{total_count}] LangGraph 파이프라인 빌드 중...")
lg_graph = build_graph(lc_retriever, qa_llm, span_extractor_fn, GRAPH_SOP_THRESHOLD,
                       checkpointer=MemorySaver())
print(f"[{now_count}/{total_count}] LangGraph 파이프라인 빌드 완료.")

now_count += 1
print(f"[{now_count}/{total_count}] Claude LangGraph 파이프라인 빌드 중...")
claude_graph = build_claude_graph(lc_retriever, GRAPH_CLAUDE_THRESHOLD,
                                  checkpointer=MemorySaver())
print(f"[{now_count}/{total_count}] Claude LangGraph 파이프라인 빌드 완료.")

print("=" * 40)
print("서버 준비 완료. 요청을 받을 수 있습니다.")
print("=" * 40)
