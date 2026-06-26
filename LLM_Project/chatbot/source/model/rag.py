import os
import json
import urllib.request

from bpe import decompose

KORQUAD_URL = "https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json"
CHUNK_LEN = 90  # block_size=128 토큰 예산 안에서 질문+참고 청크가 한 윈도우에 들어가도록 제한


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
