import json
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 코사인 유사도가 음수로 나올 수 있는 건 정상이라, langchain의 [0,1] 범위 검증 경고는 끈다 —
# 우리는 이 값을 그대로 쓰지 않고 calibrate()의 고정-보정 정규화로 다시 스케일링한다.
warnings.filterwarnings("ignore", message="Relevance scores must be between")

SOURCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_DIR / "model" / "sop_model"))  # tokenizer.py용
sys.path.insert(0, str(SOURCE_DIR))             # rag 패키지용

from rag import load_korquad_qa_pairs, chunk_context, load_ragdata_passages  # noqa: E402
from tokenizer import load_korean_chatbot_data  # noqa: E402

EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
RERANK_MODEL_NAME = "bongsoo/klue-cross-encoder-v1"  # 한국어 특화 Cross-Encoder
CALIBRATION_SAMPLE_SIZE = 80  # 보정(calibration)에 쓸 relevant/irrelevant 질문 샘플 개수
TOP_K = 5        # BM25Retriever / FAISS retriever가 EnsembleRetriever용으로 반환할 후보 개수
RERANK_TOP_K = 10  # re-rank 전 hybrid에서 뽑을 후보 수

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
PASSAGES_CACHE = CACHE_DIR / "passages.pkl"
FAISS_CACHE    = CACHE_DIR / "faiss_index"
CALIBRATION_CACHE = CACHE_DIR / "calibration.json"


class HybridRetriever:
    """BM25Retriever(sparse) + FAISS(dense)를 LangChain EnsembleRetriever로 결합한 검색기.

    EnsembleRetriever는 RRF(rank fusion)라 질문 간에 비교 가능한 절대 점수를 안 주기 때문에,
    /chat의 라우팅 임계값 판단(best_match)에는 BM25/FAISS의 원시 점수를 직접 꺼내 기존
    rag/__init__.py의 고정-보정(calibrate) 정규화 방식을 그대로 적용한다."""

    def __init__(self, passages: list[str], alpha: float = 0.5) -> None:
        self.passages = passages
        self.alpha = alpha
        self.norm_bounds = None
        self._idx_by_passage = {p: i for i, p in enumerate(passages)}

        print(f"[hybrid] Cross-Encoder 로드 중: {RERANK_MODEL_NAME}")
        self.cross_encoder = CrossEncoder(RERANK_MODEL_NAME)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )
        self.bm25 = BM25Retriever.from_texts(
            passages, k=TOP_K,
            preprocess_func=lambda t: t.lower().split(),
        )
        # langchain_community FAISS의 COSINE distance_strategy는 IndexFlatL2를 그대로 쓰고
        # normalize_L2를 무시하는 미구현 상태라, 정규화된 벡터 + 기본 EUCLIDEAN 전략을 써서
        # 코사인 유사도와 단조 대응되는 점수(_euclidean_relevance_score_fn)를 얻는다.
        if FAISS_CACHE.exists():
            print("[hybrid] FAISS 캐시 로드 중...")
            self.vectorstore = FAISS.load_local(str(FAISS_CACHE), embeddings, allow_dangerous_deserialization=True)
        else:
            # FAISS.from_texts(전체)는 M2 Apple Silicon에서 토크나이저 멀티프로세싱 segfault 유발.
            # 500개 단위 배치로 나눠서 add_texts()로 추가하는 방식으로 우회한다.
            BATCH = 500
            print(f"[hybrid] FAISS 인덱스 배치 빌드 중 (배치 {BATCH}개씩, 최초 1회)...")
            self.vectorstore = FAISS.from_texts(passages[:BATCH], embeddings, normalize_L2=True)
            for i in range(BATCH, len(passages), BATCH):
                self.vectorstore.add_texts(passages[i:i+BATCH])
                print(f"[hybrid] [{min(i+BATCH, len(passages)):,}/{len(passages):,}] 완료", flush=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(FAISS_CACHE))
            print("[hybrid] FAISS 인덱스 저장 완료.")
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        self.ensemble = EnsembleRetriever(retrievers=[self.bm25, faiss_retriever], weights=[alpha, 1 - alpha])

    def retrieve(self, query: str, top_k: int = 1) -> list[str]:
        """query와 가장 유사한 청크 top_k개를 반환 (EnsembleRetriever의 RRF 결합 순위 기준)."""
        return [doc.page_content for doc in self.ensemble.invoke(query)[:top_k]]

    def _raw_scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """전체 passages에 대한 (sparse, dense) 원시 점수 배열 — calibrate/best_match 계산용.

        FAISS는 상위 200개만 가져온다. 전체(36K)를 반환하면 SOP_GPT 모델과 메모리를 공유하는
        uvicorn 프로세스에서 OOM으로 죽는다. 200위 밖 문서의 dense 점수는 0으로 처리하며,
        그런 문서가 최적 결과가 될 가능성은 낮아 품질 영향이 거의 없다.
        """
        sparse = np.asarray(self.bm25.vectorizer.get_scores(self.bm25.preprocess_func(query)))
        dense = np.zeros(len(self.passages))
        for doc, score in self.vectorstore.similarity_search_with_relevance_scores(query, k=200):
            idx = self._idx_by_passage.get(doc.page_content)
            if idx is not None:
                dense[idx] = score
        return sparse, dense

    @staticmethod
    def _load_ragdata_questions(ragdata_dir: str | None = None) -> list[str]:
        """ragdata 마크다운 헤딩(##, ###)을 질문 형태로 추출."""
        path = Path(ragdata_dir) if ragdata_dir else CACHE_DIR.parent.parent / "ragdata"
        questions = []
        for md_file in path.rglob("*.md"):
            for line in md_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("## ") or line.startswith("### "):
                    q = line.lstrip("#").strip()
                    if q and not q.endswith("|"):  # 테이블 헤더 제외
                        questions.append(q if q.endswith("?") else q + "?")
        return questions

    def calibrate(self, root_dir: str | None = None, sample_size: int = CALIBRATION_SAMPLE_SIZE) -> None:
        """relevant(KorQuAD + ragdata 질문) / irrelevant(잡담 챗봇 질문) 샘플로 sparse/dense 점수의
        "정상 범위"를 한 번만 측정해서 고정 정규화 기준(lo, hi)을 만든다."""
        if CALIBRATION_CACHE.exists():
            print("[hybrid] calibration 캐시 로드 중...")
            self.norm_bounds = json.loads(CALIBRATION_CACHE.read_text())
            return

        print("[hybrid] calibration 계산 중... (최초 1회)")
        korquad_qs = [q for _, q, _, _ in load_korquad_qa_pairs(root_dir)]
        ragdata_qs = self._load_ragdata_questions()
        print(f"[hybrid] relevant 질문: KorQuAD {len(korquad_qs):,}개 + ragdata {len(ragdata_qs):,}개")

        # KorQuAD와 ragdata를 50:50으로 샘플링 — ragdata 도메인 쿼리가 calibration에 충분히 반영되도록.
        half = sample_size // 2
        random.Random(0).shuffle(korquad_qs)
        random.Random(1).shuffle(ragdata_qs)
        rel_sample = korquad_qs[:half] + (ragdata_qs[:half] if ragdata_qs else korquad_qs[half:sample_size])

        chatbot_text = load_korean_chatbot_data()
        irrelevant_qs = [line.split("\t")[0] for line in chatbot_text.split("\n") if "\t" in line]
        random.Random(2).shuffle(irrelevant_qs)
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
            "sparse_lo": float(irr_sparse.mean()), "sparse_hi": float(rel_sparse.mean()),
            "dense_lo": float(irr_dense.mean()), "dense_hi": float(rel_dense.mean()),
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CALIBRATION_CACHE.write_text(json.dumps(self.norm_bounds))
        print("[hybrid] calibration 캐시 저장 완료.")

    @staticmethod
    def _fixed_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(x)

    def _hybrid_scores(self, query: str) -> np.ndarray:
        if self.norm_bounds is None:
            raise RuntimeError("calibrate()가 호출되지 않았습니다. build_hybrid_retriever()를 사용하세요.")
        sparse, dense = self._raw_scores(query)
        norm_sparse = self._fixed_normalize(sparse, self.norm_bounds["sparse_lo"], self.norm_bounds["sparse_hi"])
        # dense calibration이 실패한 경우(hi==lo==0) sparse만 사용
        if self.norm_bounds["dense_hi"] <= self.norm_bounds["dense_lo"]:
            return norm_sparse
        norm_dense = self._fixed_normalize(dense, self.norm_bounds["dense_lo"], self.norm_bounds["dense_hi"])
        return self.alpha * norm_sparse + (1 - self.alpha) * norm_dense

    def best_match(self, query: str, rerank_k: int = RERANK_TOP_K) -> tuple[str, float]:
        """hybrid 점수가 가장 높은 패시지를 반환.

        Cross-Encoder 재정렬은 짧은 영어 약어 쿼리에서 오히려 품질을 낮추는 문제가 있어 제거.
        """
        scores = self._hybrid_scores(query)
        best_idx = int(scores.argmax())
        return self.passages[best_idx], float(scores[best_idx])


def build_hybrid_retriever(root_dir: str | None = None, ragdata_dir: str | None = None, alpha: float = 0.3) -> HybridRetriever:
    if PASSAGES_CACHE.exists():
        print("[hybrid] passages 캐시 로드 중...")
        passages = pickle.loads(PASSAGES_CACHE.read_bytes())
    else:
        print("[hybrid] passages 빌드 중... (최초 1회)")
        pairs = load_korquad_qa_pairs(root_dir, include_train=True)
        contexts = {context for context, _, _, _ in pairs}  # 문단 중복 제거
        passages = []
        for context in contexts:
            passages.extend(chunk_context(context))
        passages.extend(load_ragdata_passages(ragdata_dir))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PASSAGES_CACHE.write_bytes(pickle.dumps(passages))
        print("[hybrid] passages 캐시 저장 완료.")

    retriever = HybridRetriever(passages, alpha=alpha)
    retriever.calibrate(root_dir)
    return retriever
