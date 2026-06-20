import torch

from model import device, block_size
from bpe import tokenize, decode, tokenize_with_offsets, decompose, compose
import rag

def chat(model, stoi, itos, merges, base_set):
    model.load_state_dict(torch.load("SOP_GPT.pt", map_location=device))
    model.eval()
    # "." / "?" / "!"로 끝나는 토큰(예: "요.", "습니다.", "다.")이 생성되면 한 문장이 끝난 것으로 보고 멈춘다.
    stop_tokens = {i for t, i in stoi.items() if t and t[-1] in ".?!"}
    print("Type a prompt and the model will continue it (empty line or Ctrl-D to quit).")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if not prompt:
            break
        ids = [stoi[t] for t in tokenize(prompt, merges, base_set) if t in stoi]
        if not ids:
            print("(no tokens from the prompt are in the vocabulary)")
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, 200, stop_tokens=stop_tokens, temperature=0.7, top_p=0.9, repetition_penalty=1.3)[0].tolist()
        print(decode(out[len(ids):], itos))  # print only the continuation


def chat_qa(model, stoi, itos, merges, base_set):
    """Stage 2: "질문: ...\\n답변: " 포맷으로 fine-tuning된 모델과의 Q&A REPL."""
    model.load_state_dict(torch.load("SOP_GPT_qa.pt", map_location=device))
    model.eval()
    # 줄바꿈으로 끝나는 토큰이 나오면 "답변: ..." 한 줄이 끝난 것으로 보고 멈춘다.
    stop_tokens = {i for t, i in stoi.items() if t.endswith("\n")}
    print("Ask something (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("질문: ")
        except EOFError:
            break
        if not question:
            break
        prompt = f"질문: {question}\n답변: "
        ids = [stoi[t] for t in tokenize(prompt, merges, base_set) if t in stoi]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, 60, stop_tokens=stop_tokens, temperature=0.8, top_k=40, repetition_penalty=1.3)[0].tolist()
        print("답변:", decode(out[len(ids):], itos).strip())


def extract_answer(model, stoi, merges, base_set, question, context):
    """Stage 4(추출형): "질문: ...\\n참고: {context}" 안에서 정답의 시작/끝 토큰 위치를 분류해 그 구간을 그대로 잘라 반환한다.
    생성이 아니라 위치 분류라서 자기회귀 오류가 누적되지 않는다."""
    prefix = f"질문: {question}\n참고: "
    prompt = prefix + context
    tokens, offsets, decomposed = tokenize_with_offsets(prompt, merges, base_set)
    ids = [stoi[t] for t in tokens][:block_size]
    offsets = offsets[:block_size]

    nfd_prefix_len = len(decompose(prefix))
    context_token_start = next((i for i, (s, e) in enumerate(offsets) if e > nfd_prefix_len), 0)

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    start_logits, end_logits, _ = model(idx)
    start_logits, end_logits = start_logits[0], end_logits[0]
    start_logits[:context_token_start] = float("-inf")  # "질문:" 구간은 정답 후보에서 제외
    end_logits[:context_token_start] = float("-inf")

    start_idx = torch.argmax(start_logits).item()
    end_logits[:start_idx] = float("-inf")  # 끝은 시작 이후에서만 찾는다
    end_idx = torch.argmax(end_logits).item()

    char_start, char_end = offsets[start_idx][0], offsets[end_idx][1]
    return compose(decomposed[char_start:char_end])


def chat_span(model, stoi, merges, base_set, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, passages):
    """Stage 4: 질문과 가장 유사한 KorQuAD 청크를 검색해, 그 안에서 정답 구간을 직접 추출하는 RAG REPL."""
    model.load_state_dict(torch.load("SOP_GPT_span.pt", map_location=device))
    model.eval()
    print("Ask something (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("질문: ")
        except EOFError:
            break
        if not question:
            break
        context = rag.retrieve(question, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, passages, top_k=1)[0]
        answer = extract_answer(model, stoi, merges, base_set, question, context)
        print("참고:", context)
        print("답변:", answer)
