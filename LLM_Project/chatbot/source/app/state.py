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

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

BPE_PATH  = MODEL_DIR / "bpe_vocab.json"
GEN_CKPT  = MODEL_DIR / "SOP_GPT.pt"
QA_CKPT   = MODEL_DIR / "SOP_GPT_qa.pt"
SPAN_CKPT = MODEL_DIR / "SOP_GPT_span.pt"

RAG_SIM_THRESHOLD   = 0.515   # BM25+FAISS 하이브리드 held-out 검증 최적값
TFIDF_SIM_THRESHOLD = 0.25    # TF-IDF 단독 임계값

# ── BPE 토크나이저 ─────────────────────────────────────────────────────────────
print("[1/6] BPE 토크나이저 로딩 중...")
vocab, merges = load_bpe(BPE_PATH)
stoi, itos    = build_vocab(vocab)
vocab_size    = len(vocab)
base_set      = base_alphabet(vocab)
print("[1/6] BPE 토크나이저 로딩 완료.")

# ── PyTorch 모델 ───────────────────────────────────────────────────────────────
print("[2/6] 이어쓰기 모델(gen) 로딩 중...")
gen_model = SOP_GPT(vocab_size).to(device)
gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
gen_model.eval()
print("[2/6] 이어쓰기 모델(gen) 로딩 완료.")

print("[3/6] QA 모델 로딩 중...")
qa_model = SOP_GPT(vocab_size).to(device)
qa_model.load_state_dict(torch.load(QA_CKPT, map_location=device))
qa_model.eval()
print("[3/6] QA 모델 로딩 완료.")

print("[4/6] Span 추출 모델 로딩 중...")
span_model = SOP_GPT_Span(vocab_size).to(device)
span_model.load_state_dict(torch.load(SPAN_CKPT, map_location=device))
span_model.eval()
print("[4/6] Span 추출 모델 로딩 완료.")

# ── 검색기 ─────────────────────────────────────────────────────────────────────
print("[5/6] TF-IDF 검색기 로딩 중...")
tfidf_retriever = build_tfidf_retriever()
print("[5/6] TF-IDF 검색기 로딩 완료.")

print("[6/6] 하이브리드 검색기(BM25+FAISS) 로딩 중...")
lc_retriever = build_hybrid_retriever()
print("[6/6] 하이브리드 검색기(BM25+FAISS) 로딩 완료.")

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

print("=" * 40)
print("서버 준비 완료. 요청을 받을 수 있습니다.")
print("=" * 40)
