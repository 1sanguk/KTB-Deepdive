import os
import csv
import urllib.request

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


def load_chatbot_qa_data(root_dir=None):
    """Q&A 쌍을 "질문: ...\\n답변: ...\\n\\n" 포맷으로 만들어 합친다 (Stage 2 fine-tuning용)."""
    local_path = download_korean_chatbot_data(root_dir)
    chunks = []
    with open(local_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            question, answer, _label = row
            chunks.append(f"질문: {question}\n답변: {answer}\n\n")
    return "".join(chunks)


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
