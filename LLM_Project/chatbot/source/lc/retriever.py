import random
import sys
import warnings
from pathlib import Path

import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 코사인 유사도가 음수로 나올 수 있는 건 정상이라, langchain의 [0,1] 범위 검증 경고는 끈다 —
# 우리는 이 값을 그대로 쓰지 않고 calibrate()의 고정-보정 정규화로 다시 스케일링한다.
warnings.filterwarnings("ignore", message="Relevance scores must be between")

SOURCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_DIR / "model"))  # tokenizer.py용
sys.path.insert(0, str(SOURCE_DIR))             # rag 패키지용

from rag import load_korquad_qa_pairs, chunk_context, load_ragdata_passages  # noqa: E402
from tokenizer import load_korean_chatbot_data  # noqa: E402

EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
CALIBRATION_SAMPLE_SIZE = 80  # 보정(calibration)에 쓸 relevant/irrelevant 질문 샘플 개수
TOP_K = 5  # BM25Retriever / FAISS retriever가 EnsembleRetriever용으로 반환할 후보 개수


class HybridRetriever:
    """BM25Retriever(sparse) + FAISS(dense)를 LangChain EnsembleRetriever로 결합한 검색기.

    EnsembleRetriever는 RRF(rank fusion)라 질문 간에 비교 가능한 절대 점수를 안 주기 때문에,
    /chat의 라우팅 임계값 판단(best_match)에는 BM25/FAISS의 원시 점수를 직접 꺼내 기존
    rag/__init__.py의 고정-보정(calibrate) 정규화 방식을 그대로 적용한다."""

    def __init__(self, passages, alpha=0.5):
        self.passages = passages
        self.alpha = alpha
        self.norm_bounds = None

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.bm25 = BM25Retriever.from_texts(passages, k=TOP_K)
        # langchain_community FAISS의 COSINE distance_strategy는 IndexFlatL2를 그대로 쓰고
        # normalize_L2를 무시하는 미구현 상태라, 정규화된 벡터 + 기본 EUCLIDEAN 전략을 써서
        # 코사인 유사도와 단조 대응되는 점수(_euclidean_relevance_score_fn)를 얻는다.
        self.vectorstore = FAISS.from_texts(passages, embeddings, normalize_L2=True)
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        self.ensemble = EnsembleRetriever(retrievers=[self.bm25, faiss_retriever], weights=[alpha, 1 - alpha])

    def retrieve(self, query, top_k=1):
        """query와 가장 유사한 청크 top_k개를 반환 (EnsembleRetriever의 RRF 결합 순위 기준)."""
        return [doc.page_content for doc in self.ensemble.invoke(query)[:top_k]]

    def _raw_scores(self, query):
        """전체 passages에 대한 (sparse, dense) 원시 점수 배열 — calibrate/best_match 계산용."""
        sparse = np.asarray(self.bm25.vectorizer.get_scores(self.bm25.preprocess_func(query)))
        dense = np.zeros(len(self.passages))
        idx_by_passage = {p: i for i, p in enumerate(self.passages)}
        for doc, score in self.vectorstore.similarity_search_with_relevance_scores(query, k=len(self.passages)):
            dense[idx_by_passage[doc.page_content]] = score
        return sparse, dense

    def calibrate(self, root_dir=None, sample_size=CALIBRATION_SAMPLE_SIZE):
        """relevant(KorQuAD 질문) / irrelevant(잡담 챗봇 질문) 샘플로 sparse/dense 점수의
        "정상 범위"를 한 번만 측정해서 고정 정규화 기준(lo, hi)을 만든다."""
        relevant_qs = [q for _, q, _, _ in load_korquad_qa_pairs(root_dir)]
        chatbot_text = load_korean_chatbot_data()
        irrelevant_qs = [line.split("\t")[0] for line in chatbot_text.split("\n") if "\t" in line]

        random.Random(0).shuffle(relevant_qs)
        random.Random(1).shuffle(irrelevant_qs)
        rel_sample = relevant_qs[:sample_size]
        irr_sample = irrelevant_qs[:sample_size]

        def raw_best_scores(qs):
            sparse, dense = [], []
            for q in qs:
                s, d = self._raw_scores(q)
                sparse.append(s.max())
                dense.append(d.max())
            return np.array(sparse), np.array(dense)

        rel_sparse, rel_dense = raw_best_scores(rel_sample)
        irr_sparse, irr_dense = raw_best_scores(irr_sample)
        self.norm_bounds = {
            "sparse_lo": irr_sparse.mean(), "sparse_hi": rel_sparse.mean(),
            "dense_lo": irr_dense.mean(), "dense_hi": rel_dense.mean(),
        }

    @staticmethod
    def _fixed_normalize(x, lo, hi):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(x)

    def _hybrid_scores(self, query):
        sparse, dense = self._raw_scores(query)
        norm_sparse = self._fixed_normalize(sparse, self.norm_bounds["sparse_lo"], self.norm_bounds["sparse_hi"])
        norm_dense = self._fixed_normalize(dense, self.norm_bounds["dense_lo"], self.norm_bounds["dense_hi"])
        return self.alpha * norm_sparse + (1 - self.alpha) * norm_dense

    def best_match(self, query):
        """가장 유사한 청크와 그 하이브리드 점수를 함께 반환 (관련 문서가 없을 때 폴백 판단용)."""
        scores = self._hybrid_scores(query)
        best_idx = scores.argmax()
        return self.passages[best_idx], float(scores[best_idx])


def build_hybrid_retriever(root_dir=None, ragdata_dir=None):
    pairs = load_korquad_qa_pairs(root_dir, include_train=True)
    contexts = {context for context, _, _, _ in pairs}  # 문단 중복 제거
    passages = []
    for context in contexts:
        passages.extend(chunk_context(context))
    passages.extend(load_ragdata_passages(ragdata_dir))

    retriever = HybridRetriever(passages)
    retriever.calibrate(root_dir)
    return retriever
