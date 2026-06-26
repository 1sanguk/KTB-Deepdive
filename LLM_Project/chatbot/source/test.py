"""Colab 등 외부 환경에서 LangChain 기반 RAG 검색기(HybridRetriever)가
의도대로 동작하는지 한 파일로 확인하는 스모크 테스트.

Colab 사용법:
    !pip install langchain langchain-community langchain-huggingface faiss-cpu rank_bm25
    !git clone <repo_url> chatbot
    %cd chatbot/source
    !python test.py

주의: SOP_GPT 체크포인트(.pt)는 git에 들어있지만 bpe_vocab.json은 들어있지 않아서
전체 FastAPI /chat 엔드포인트(생성 모델 포함)는 이 스크립트로 검증할 수 없다.
여기서는 모델 로딩 없이도 동작하는 검색/라우팅 로직(HybridRetriever.best_match)만 확인한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_rag.retriever import build_hybrid_retriever  # noqa: E402
from rag import load_korquad_qa_pairs  # noqa: E402

RAG_SIM_THRESHOLD = 0.515  # app.py와 동일한 라우팅 임계값 (하이브리드+고정보정, held-out 정확도 82.7%)
N_KORQUAD_SAMPLES = 8  # 실제 KorQuAD 질문 샘플 (관련 문서가 있는 경우)
RANDOM_SEED = 2

# 임의로 지어낸 일반 상식 질문은 KorQuAD dev set 청크와 우연히 안 겹칠 수 있어 검증용으로 부적합하다
# (90자 고정 청크라 질문과 무관하게 잘려서, 지어낸 질문은 실제로 매칭될 청크가 없을 수 있음) —
# 그래서 "관련 있음" 케이스는 데이터셋에 실제로 들어있는 질문에서 뽑는다.
CHITCHAT_QUESTIONS = ["오늘 기분 어때?", "취업 준비 잘하고 있어?", "사랑해"]


def main():
    print("Building hybrid retriever (KorQuAD 인덱싱 + 임베딩 모델 다운로드, 1~2분 소요)...")
    retriever = build_hybrid_retriever()

    import random
    pairs = load_korquad_qa_pairs()
    sample = random.Random(RANDOM_SEED).sample(pairs, N_KORQUAD_SAMPLES)
    test_cases = [(question, True) for _, question, _, _ in sample]
    test_cases += [(q, False) for q in CHITCHAT_QUESTIONS]

    correct = 0
    for question, expect_rag in test_cases:
        context, score = retriever.best_match(question)
        used_rag = score >= RAG_SIM_THRESHOLD
        ok = used_rag == expect_rag
        correct += ok
        print(f"\n[{'OK' if ok else 'MISMATCH'}] '{question}'")
        print(f"  score={score:.3f}  used_rag={used_rag} (expected {expect_rag})")
        print(f"  context: {context[:80]}...")

    print(f"\n{correct}/{len(test_cases)} 라우팅 기대값과 일치")


if __name__ == "__main__":
    main()
