import io
import json
import os
import csv
import re
import urllib.request
import zipfile
from pathlib import Path

# --- data: Fetched from Korpora ---

def download_korean_chatbot_data(root_dir=None, force_download=False):
    if root_dir is None:
        root_dir = os.path.join(os.path.expanduser("~"), ".korpora")
    dataset_dir = os.path.join(root_dir, "korean_chatbot_data")
    os.makedirs(dataset_dir, exist_ok=True)
    local_path = os.path.join(dataset_dir, "ChatbotData.csv")
    if force_download or not os.path.exists(local_path):
        url = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
        print("Downloading Korean chatbot data from", url)
        urllib.request.urlretrieve(url, local_path)
    return local_path


def load_korean_chatbot_data(root_dir=None):
    local_path = download_korean_chatbot_data(root_dir)
    questions, answers = [], []
    with open(local_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            question, answer, _label = row
            questions.append(question)
            answers.append(answer)
    return "\n".join(q + "\t" + a for q, a in zip(questions, answers))


def load_chatbot_qa_pairs(root_dir=None):
    """(question, answer) 튜플 리스트를 반환 (Stage 2 / DPO 학습 공용)."""
    local_path = download_korean_chatbot_data(root_dir)
    pairs = []
    with open(local_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            question, answer, _label = row
            if question.strip() and answer.strip():
                pairs.append((question.strip(), answer.strip()))
    return pairs


def load_chatbot_qa_data(root_dir=None):
    """Q&A 쌍을 "질문: ...\\n답변: ...<|endoftext|>\\n" 포맷으로 반환 (Stage 2 fine-tuning용)."""
    from bpe import EOS_TOKEN
    return "".join(
        f"질문: {q}\n답변: {a}{EOS_TOKEN}\n"
        for q, a in load_chatbot_qa_pairs(root_dir)
    )


AIHUB_DIR = Path.home() / "Korpora" / "aihub"


_NOISE_RE = re.compile(r'^[\s\d\W]+$')  # 숫자·특수문자·공백만 있는 줄


def _clean_utterance(line: str) -> str:
    """발화 텍스트를 정제한다. 너무 짧거나 노이즈인 줄은 빈 문자열로 반환."""
    line = line.strip()
    if len(line) < 5:
        return ""
    if _NOISE_RE.match(line):
        return ""
    # 중복 공백 정리
    return re.sub(r' {2,}', ' ', line)


def _read_txt_conversation(text: str) -> str:
    """'화자번호 : 발화' 형식의 txt를 대화 텍스트로 변환."""
    seen, lines = set(), []
    for line in text.splitlines():
        m = re.match(r'^\d+\s*:\s*(.+)', line.strip())
        utt = _clean_utterance(m.group(1) if m else line)
        if utt and utt not in seen:
            seen.add(utt)
            lines.append(utt)
    return "\n".join(lines)


def _read_json_conversation(data: dict) -> str:
    """AI Hub JSON 형식에서 body[].utterance를 추출해 대화 텍스트로 변환."""
    body = data.get("body", [])
    seen, lines = set(), []
    for entry in body:
        utt = _clean_utterance(entry.get("utterance", ""))
        if utt and utt not in seen:
            seen.add(utt)
            lines.append(utt)
    return "\n".join(lines)


def load_aihub_conversation_data(aihub_dir=None, training_only=True, max_chars=None) -> str:
    """~/Korpora/aihub/ 하위 txt/json 파일에서 대화 텍스트를 읽어 하나의 문자열로 반환.

    training_only=True이면 파일 경로에 'Training' 또는 '1.Training'이 포함된 파일만 사용.
    max_chars: 이 글자 수에 도달하면 파일 읽기를 중단 (None이면 전체 읽음).
    """
    base = Path(aihub_dir) if aihub_dir else AIHUB_DIR
    if not base.exists():
        print(f"[aihub] 폴더 없음: {base}")
        return ""

    chunks = []
    total_chars = 0
    files = sorted(f for f in base.rglob("*") if f.suffix in (".txt", ".json") and f.is_file())
    print(f"[aihub] 파일 {len(files):,}개 발견")

    for file_path in files:
        if max_chars and total_chars >= max_chars:
            break
        path_str = str(file_path)
        if "라벨링" in path_str:
            continue
        if training_only and not any(k in path_str for k in ["Training", "1.Training", "train"]):
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if file_path.suffix == ".txt":
                conv = _read_txt_conversation(text)
            else:
                conv = _read_json_conversation(json.loads(text))
            if conv:
                chunks.append(conv)
                total_chars += len(conv)
        except Exception as e:
            print(f"[aihub] 오류 {file_path.name}: {e}")

    result = "\n\n".join(chunks)
    print(f"[aihub] {len(chunks):,}개 대화, {len(result):,}자 로드 완료")
    return result


def download_kowikitext_data(root_dir=None, force_download=False):
    from Korpora.korpus_kowiki import KOWIKI_FETCH_INFORMATION
    from Korpora.utils import fetch

    if root_dir is None:
        root_dir = os.path.join(os.path.expanduser("~"), ".korpora")

    local_paths = []
    for info in KOWIKI_FETCH_INFORMATION:
        local_path = os.path.join(os.path.abspath(root_dir), info["destination"])
        fetch(info["url"], local_path, "kowikitext", force_download, info["method"])
        local_paths.append(local_path[:-len(".zip")])
    return local_paths


def load_kowikitext_data(root_dir=None, train_chars=0):
    """kowikitext dev+test 전체와, train의 앞부분 `train_chars`자(0이면 제외)를 합쳐서 반환.

    train split은 압축 해제 시 ~1.6GB라 전체를 쓰면 너무 크므로 앞부분만 잘라서 사용한다.
    """
    local_paths = download_kowikitext_data(root_dir)
    texts = []
    for path in local_paths:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".train"):
                if train_chars:
                    texts.append(f.read(train_chars))
            else:
                texts.append(f.read())
    return "\n".join(texts)
