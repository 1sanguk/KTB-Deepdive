import os
import json
import random
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from bpe import decompose
from tokenizer import load_korean_chatbot_data

CALIBRATION_SAMPLE_SIZE = 80  # 보정(calibration)에 쓸 relevant/irrelevant 질문 샘플 개수

KORQUAD_URL = "https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json"
CHUNK_LEN = 90  # block_size=128 토큰 예산 안에서 질문+참고 청크가 한 윈도우에 들어가도록 제한

# 실제 SKT KoBERT는 별도 sentencepiece 토크나이저 패키지가 필요해서, transformers의
# AutoTokenizer/AutoModel로 바로 쓸 수 있는 동급의 한국어 임베딩 모델로 대체했다.
# (TF-IDF는 단어가 같아야만 매칭되는데, "게임 개발자"와 "디지털화폐 개발자"를 "개발자"라는
# 글자만 보고 같다고 오검색하는 문제가 있었음 — 문맥을 이해하는 임베딩으로 바꿔 완화)
# klue/bert-base(일반 MLM 사전학습)는 문장 유사도용으로 학습되지 않아 모든 문장이
# 비슷한 점수로 뭉치는 anisotropy 문제가 있어서, 문장 유사도(STS)로 파인튜닝된
# ko-sroberta로 교체 — 코사인 유사도 기반 검색에 훨씬 적합하다.
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
_embed_tokenizer = None
_embed_model = None


def _get_embedder():
    global _embed_tokenizer, _embed_model
    if _embed_model is None:
        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _embed_model.eval()
    return _embed_tokenizer, _embed_model


@torch.no_grad()
def embed(texts, batch_size=32):
    """texts(list[str]) -> (N, hidden_size) 문장 임베딩. attention mask로 평균 풀링."""
    tokenizer, model = _get_embedder()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
        hidden = model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(pooled)
    return torch.cat(vectors, dim=0)


def download_korquad_data(root_dir=None, force_download=False):
    if root_dir is None:
        root_dir = os.path.join(os.path.expanduser("~"), ".korpora")
    dataset_dir = os.path.join(root_dir, "korquad")
    os.makedirs(dataset_dir, exist_ok=True)
    local_path = os.path.join(dataset_dir, "KorQuAD_v1.0_dev.json")
    if force_download or not os.path.exists(local_path):
        print("Downloading KorQuAD v1.0 dev set from", KORQUAD_URL)
        urllib.request.urlretrieve(KORQUAD_URL, local_path)
    return local_path


def load_korquad_qa_pairs(root_dir=None):
    """KorQuAD json -> (context, question, answer_text, answer_start) 리스트."""
    local_path = download_korquad_data(root_dir)
    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if not qa["answers"]:
                    continue
                answer = qa["answers"][0]
                pairs.append((context, qa["question"], answer["text"], answer["answer_start"]))
    return pairs


def chunk_context(context, chunk_len=CHUNK_LEN):
    """문단을 질문과 무관하게 겹치지 않는 chunk_len자 청크로 분할 (검색 인덱스용)."""
    return [context[i:i + chunk_len] for i in range(0, len(context), chunk_len) if context[i:i + chunk_len].strip()]


def answer_window(context, answer_start, answer_text, chunk_len=CHUNK_LEN):
    """정답 중심 chunk_len자 윈도우(snippet)와, 그 윈도우 안에서 정답이 차지하는
    로컬 문자 위치 [local_start, local_end)를 함께 반환한다 (추출형 QA 학습 데이터용)."""
    answer_end = answer_start + len(answer_text)
    center = (answer_start + answer_end) // 2
    half = chunk_len // 2
    start = max(0, center - half)
    end = min(len(context), start + chunk_len)
    start = max(0, end - chunk_len)  # 문단 끝에 가까우면 시작점을 당겨서 chunk_len 길이를 최대한 채움
    snippet = context[start:end]
    local_start = max(0, answer_start - start)
    local_end = min(len(snippet), answer_end - start)
    return snippet, local_start, local_end


def answer_centered_chunk(context, answer_start, answer_text, chunk_len=CHUNK_LEN):
    """정답이 포함되도록 정답 주변 chunk_len자 윈도우만 잘라낸다 (snippet 텍스트만 필요할 때)."""
    snippet, _, _ = answer_window(context, answer_start, answer_text, chunk_len)
    return snippet


def build_span_examples(root_dir=None):
    """KorQuAD QA pair -> (prompt_text, nfd_start, nfd_end) 리스트.

    prompt_text = "질문: {질문}\\n참고: {정답중심 청크}" (NFC 원문).
    nfd_start/nfd_end는 decompose(prompt_text) 안에서 정답이 차지하는 [start, end) 오프셋 —
    NFD는 글자 단위로 독립적으로 적용되므로 prefix 길이를 그대로 누적해서 계산할 수 있다.
    """
    pairs = load_korquad_qa_pairs(root_dir)
    examples = []
    for context, question, answer_text, answer_start in pairs:
        snippet, local_start, local_end = answer_window(context, answer_start, answer_text)
        if local_start >= local_end:
            continue  # 윈도우가 정답을 온전히 담지 못한 경우 스킵

        prefix = f"질문: {question}\n참고: "
        prompt = prefix + snippet
        nfd_prefix_len = len(decompose(prefix))
        nfd_start = nfd_prefix_len + len(decompose(snippet[:local_start]))
        nfd_end = nfd_prefix_len + len(decompose(snippet[:local_end]))
        examples.append((prompt, nfd_start, nfd_end))
    return examples


def build_index(root_dir=None):
    """검색 인덱스(고유 문단 -> 청크 목록) + TF-IDF 벡터라이저 + 임베딩 행렬 + 정규화 보정값을 만든다.

    TF-IDF 단독은 단어만 같으면 오검색하고("게임 개발자" vs "디지털화폐 개발자"),
    임베딩 단독은 격식체/구어체처럼 문체가 비슷하면 주제가 달라도 점수가 높게 나온다 —
    두 신호를 정규화해서 평균 낸 하이브리드 점수로 서로의 약점을 보완한다."""
    pairs = load_korquad_qa_pairs(root_dir)
    contexts = {context for context, _, _, _ in pairs}  # 문단 중복 제거

    passages = []
    for context in contexts:
        passages.extend(chunk_context(context))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
    embed_matrix = embed(passages)
    norm_bounds = calibrate(tfidf_vectorizer, tfidf_matrix, embed_matrix, root_dir)
    return tfidf_vectorizer, tfidf_matrix, embed_matrix, passages, norm_bounds


def calibrate(tfidf_vectorizer, tfidf_matrix, embed_matrix, root_dir=None, sample_size=CALIBRATION_SAMPLE_SIZE):
    """relevant(KorQuAD 질문) / irrelevant(잡담 챗봇 질문) 샘플로 sparse/dense 점수의
    "정상 범위"를 한 번만 측정해서 고정 정규화 기준(lo, hi)을 만든다.

    질문마다 그 질문 안에서만 min-max를 다시 계산하면, 1등 청크는 관련 있든 없든 항상
    1.0 근처로 나와버려서 "이 질문이 애초에 관련이 있는가"를 판단할 수 없다 — relevant/irrelevant
    질문 집합 전체를 기준으로 한 번 고정해야 질문들 사이에 비교 가능한 절대적인 점수가 된다.
    """
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
            sparse.append(cosine_similarity(tfidf_vectorizer.transform([q]), tfidf_matrix)[0].max())
            dense.append(F.cosine_similarity(embed([q]), embed_matrix).numpy().max())
        return np.array(sparse), np.array(dense)

    rel_sparse, rel_dense = raw_best_scores(rel_sample)
    irr_sparse, irr_dense = raw_best_scores(irr_sample)
    return {
        "sparse_lo": irr_sparse.mean(), "sparse_hi": rel_sparse.mean(),
        "dense_lo": irr_dense.mean(), "dense_hi": rel_dense.mean(),
    }


def _fixed_normalize(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(x)


def _hybrid_scores(query, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, alpha=0.5):
    tfidf_sims = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_matrix)[0]
    embed_sims = F.cosine_similarity(embed([query]), embed_matrix).numpy()
    norm_sparse = _fixed_normalize(tfidf_sims, norm_bounds["sparse_lo"], norm_bounds["sparse_hi"])
    norm_dense = _fixed_normalize(embed_sims, norm_bounds["dense_lo"], norm_bounds["dense_hi"])
    return alpha * norm_sparse + (1 - alpha) * norm_dense


def retrieve(query, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, passages, top_k=1):
    """query와 가장 유사한 청크 top_k개를 반환 (TF-IDF + 임베딩 하이브리드 점수 기준)."""
    scores = _hybrid_scores(query, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds)
    top_indices = scores.argsort()[::-1][:top_k]
    return [passages[i] for i in top_indices]


def best_match(query, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, passages):
    """가장 유사한 청크와 그 하이브리드 점수를 함께 반환 (관련 문서가 없을 때 폴백 판단용)."""
    scores = _hybrid_scores(query, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds)
    best_idx = scores.argmax()
    return passages[best_idx], float(scores[best_idx])
