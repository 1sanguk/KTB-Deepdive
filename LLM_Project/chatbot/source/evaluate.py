"""LangSmith Dataset 기반 체인 평가 스크립트.

KorQuAD dev set에서 예제를 뽑아 Dataset을 만들고,
basic / tfidf_rag / lc_rag 세 체인의 정답 포함률을 비교한다.

실행:
    cd source
    python evaluate.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "api_keys")

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from langsmith import Client
from langsmith.evaluation import evaluate

from bpe import build_vocab, base_alphabet, load_bpe
from model import device, SOP_GPT, SOP_GPT_Span
from lc.llm import SOP_GPT_LLM, make_span_extractor
from lc.chain import build_basic_chain, build_rag_chain
from lc.retriever import build_hybrid_retriever
from rag.rag import build_tfidf_retriever, load_korquad_qa_pairs
from lg.graph import build_graph

# ── 설정 ───────────────────────────────────────────────────────────────────────
MODEL_DIR        = Path(__file__).resolve().parent / "model"
DATASET_NAME     = "sop-gpt-korquad"
EVAL_SAMPLES     = 30   # 평가에 쓸 KorQuAD 예제 수 (추론이 느리므로 소규모로)
RAG_THRESHOLD    = 0.515
TFIDF_THRESHOLD  = 0.25
GRAPH_THRESHOLD  = [0.25, 0.2, 0.15]

# ── 모델 & 체인 로드 ───────────────────────────────────────────────────────────
print("모델 로딩 중...")
vocab, merges = load_bpe(MODEL_DIR / "bpe_vocab.json")
stoi, itos = build_vocab(vocab)
base_set = base_alphabet(vocab)
vocab_size = len(vocab)

qa_model = SOP_GPT(vocab_size).to(device)
qa_model.load_state_dict(torch.load(MODEL_DIR / "SOP_GPT_qa.pt", map_location=device))
qa_model.eval()

span_model = SOP_GPT_Span(vocab_size).to(device)
span_model.load_state_dict(torch.load(MODEL_DIR / "SOP_GPT_span.pt", map_location=device))
span_model.eval()

qa_llm = SOP_GPT_LLM(
    torch_model=qa_model, stoi=stoi, itos=itos, merges=merges, base_set=base_set,
    stop_on="line", temperature=0.8, top_k=40, repetition_penalty=1.3, max_new_tokens=60,
)
span_extractor_fn = make_span_extractor(span_model, stoi, merges, base_set)

print("검색기 빌드 중...")
tfidf_retriever = build_tfidf_retriever()
lc_retriever    = build_hybrid_retriever()

basic_chain     = build_basic_chain(qa_llm)
tfidf_rag_chain = build_rag_chain(tfidf_retriever, qa_llm, span_extractor_fn, TFIDF_THRESHOLD)
lc_rag_chain    = build_rag_chain(lc_retriever,    qa_llm, span_extractor_fn, RAG_THRESHOLD)
lg_graph        = build_graph(lc_retriever, qa_llm, span_extractor_fn, GRAPH_THRESHOLD)

# ── Dataset 생성 ───────────────────────────────────────────────────────────────
import os
client = Client(
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
    api_key=os.getenv("LANGSMITH_API_KEY"),
)

existing = [d.name for d in client.list_datasets()]
if DATASET_NAME not in existing:
    print(f"Dataset '{DATASET_NAME}' 생성 중...")
    pairs = load_korquad_qa_pairs()
    samples = pairs[:EVAL_SAMPLES]

    dataset = client.create_dataset(DATASET_NAME, description="KorQuAD dev set 기반 평가")
    client.create_examples(
        inputs  =[{"question": q} for _, q, _, _ in samples],
        outputs =[{"answer": a}   for _, _, a, _ in samples],
        dataset_id=dataset.id,
    )
    print(f"  {len(samples)}개 예제 업로드 완료")
else:
    print(f"Dataset '{DATASET_NAME}' 이미 존재, 재사용")

# ── Evaluator ──────────────────────────────────────────────────────────────────
def contains_match(run, example):
    """정답 텍스트가 예측 답변에 포함되어 있으면 1, 아니면 0."""
    pred     = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    score    = int(expected.strip() in pred) if expected.strip() else 0
    return {"key": "contains_match", "score": score}

# ── 각 체인 평가 ───────────────────────────────────────────────────────────────
def run_basic(inputs):
    answer = basic_chain.invoke(inputs["question"])
    return {"answer": answer}

def run_tfidf_rag(inputs):
    result = tfidf_rag_chain.invoke(inputs["question"])
    return {"answer": result["answer"]}

def run_lc_rag(inputs):
    result = lc_rag_chain.invoke(inputs["question"])
    return {"answer": result["answer"]}

def run_lg_graph(inputs):
    result = lg_graph.invoke({
        "query": inputs["question"],
        "documents": [],
        "score": 0.0,
        "answer": "",
        "used_rag": False,
        "retry_count": 0,
        "messages": [],
    })
    return {"answer": result["answer"]}

for label, target in [
    ("basic",     run_basic),
    ("tfidf_rag", run_tfidf_rag),
    ("lc_rag",    run_lc_rag),
    ("lg_graph",  run_lg_graph),
]:
    print(f"\n[{label}] 평가 중...")
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[contains_match],
        experiment_prefix=f"sop-gpt-{label}",
        client=client,
    )
    scores = [r["evaluation_results"]["results"][0].score for r in results._results]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  contains_match: {avg:.1%} ({sum(scores)}/{len(scores)})")

print("\n평가 완료. LangSmith에서 결과를 확인하세요.")
