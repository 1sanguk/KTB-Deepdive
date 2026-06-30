import os
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from bpe import decompose

KORQUAD_DEV_URL   = "https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json"
KORQUAD_TRAIN_URL = "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json"
CHUNK_LEN = 180  # block_size=256 기준: 180자 ≈ 120토큰, 질문 길이 더해도 256 이하로 맞춰짐
RAGDATA_DIR = Path(__file__).resolve().parent.parent.parent / "ragdata"


def download_korquad_data(root_dir=None, force_download=False, include_train=False):
    """dev(항상) + train(include_train=True) JSON 파일을 캐시하고 경로 리스트를 반환한다."""
    if root_dir is None:
        root_dir = os.path.join(os.path.expanduser("~"), ".korpora")
    dataset_dir = os.path.join(root_dir, "korquad")
    os.makedirs(dataset_dir, exist_ok=True)

    targets = [(KORQUAD_DEV_URL, "KorQuAD_v1.0_dev.json")]
    if include_train:
        targets.append((KORQUAD_TRAIN_URL, "KorQuAD_v1.0_train.json"))

    paths = []
    for url, fname in targets:
        local_path = os.path.join(dataset_dir, fname)
        if force_download or not os.path.exists(local_path):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, local_path)
        paths.append(local_path)
    return paths


def _parse_korquad_file(path):
    """KorQuAD JSON 파일 하나 -> (context, question, answer_text, answer_start) 리스트."""
    with open(path, encoding="utf-8") as f:
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


def load_korquad_qa_pairs(root_dir=None, include_train=False):
    """KorQuAD json -> (context, question, answer_text, answer_start) 리스트.

    include_train=True 시 train set(약 60K 쌍)까지 포함한다.
    검색 인덱스 구축에는 dev만(속도), Stage 2/4 학습에는 train 포함이 권장된다.
    """
    paths = download_korquad_data(root_dir, include_train=include_train)
    pairs = []
    for path in paths:
        pairs.extend(_parse_korquad_file(path))
    return pairs


def load_korquad_qa_data(root_dir=None):
    """KorQuAD v1.0 train+dev -> "질문: ...\\n답변: ...\\n\\n" 포맷 문자열 (Stage 2 fine-tuning용)."""
    pairs = load_korquad_qa_pairs(root_dir, include_train=True)
    print(f"KorQuAD QA pairs loaded: {len(pairs):,}")
    return "".join(f"질문: {q}\n답변: {a}\n\n" for _, q, a, _ in pairs)


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


def build_span_examples(root_dir=None, include_train=True):
    """KorQuAD QA pair -> (prompt_text, nfd_start, nfd_end) 리스트.

    prompt_text = "질문: {질문}\\n참고: {정답중심 청크}" (NFC 원문).
    nfd_start/nfd_end는 decompose(prompt_text) 안에서 정답이 차지하는 [start, end) 오프셋 —
    NFD는 글자 단위로 독립적으로 적용되므로 prefix 길이를 그대로 누적해서 계산할 수 있다.
    include_train=True(기본값)이면 train set까지 포함해 더 많은 학습 예제를 만든다.
    """
    pairs = load_korquad_qa_pairs(root_dir, include_train=include_train)
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


class TfidfRetriever:
    """scikit-learn TfidfVectorizer 기반 단순 코사인 유사도 검색기."""

    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(passages)

    def best_match(self, query):
        """가장 유사한 청크와 코사인 유사도 점수를 함께 반환."""
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        best_idx = int(scores.argmax())
        return self.passages[best_idx], float(scores[best_idx])

    def retrieve(self, query, top_k=1):
        """상위 top_k개 청크를 유사도 내림차순으로 반환."""
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.passages[i] for i in top_indices]


def load_ragdata_passages(ragdata_dir=None, chunk_len=CHUNK_LEN):
    """ragdata/ 폴더의 .txt/.md 파일을 읽어 chunk로 분할한 패시지 리스트를 반환."""
    path = Path(ragdata_dir) if ragdata_dir else RAGDATA_DIR
    if not path.exists():
        return []
    passages = []
    for f in sorted(path.rglob("*")):
        if f.suffix in (".txt", ".md") and f.is_file():
            text = f.read_text(encoding="utf-8").strip()
            if text:
                passages.extend(chunk_context(text, chunk_len))
    return passages


def build_tfidf_retriever(root_dir=None, ragdata_dir=None):
    pairs = load_korquad_qa_pairs(root_dir, include_train=True)
    contexts = {context for context, _, _, _ in pairs}
    passages = []
    for context in contexts:
        passages.extend(chunk_context(context))
    passages.extend(load_ragdata_passages(ragdata_dir))
    return TfidfRetriever(passages)
