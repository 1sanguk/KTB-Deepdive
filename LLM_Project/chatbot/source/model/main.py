import os
import random
import sys
import torch

from chat import chat, chat_qa, chat_span
from model import device, steps, lr, SOP_GPT, SOP_GPT_Span, block_size
from tokenizer import load_korean_chatbot_data, load_kowikitext_data, load_chatbot_qa_data
from bpe import train_bpe, build_vocab, base_alphabet, encode, decode, save_bpe, load_bpe, tokenize_with_offsets
from train_utils import make_batcher, train_loop
import rag

BPE_VOCAB_SIZE = 8000
BPE_PATH = "bpe_vocab.json"
KOWIKI_TRAIN_CHARS = 80_000_000  # kowikitext train split은 ~1.6GB라 앞부분만 사용 (BPE 학습용 전체 코퍼스 기준)

# Stage1 학습 코퍼스 믹스: kowikitext(위키체)가 chatbot 데이터(구어체)보다 250배 가까이 많아서
# "오늘 날씨는" 같은 구어 이어쓰기에도 위키 말투가 나오는 문제가 있었음 -> kowiki 비중을 줄이고
# chatbot 데이터를 업샘플링(반복)해서 두 말투의 비중을 좀 더 비슷하게 맞춤
STAGE1_KOWIKI_TRAIN_CHARS = 15_000_000
STAGE1_CHATBOT_UPSAMPLE = 15

GEN_CKPT = "SOP_GPT.pt"       # Stage 1(이어쓰기) 가중치
QA_CKPT = "SOP_GPT_qa.pt"     # Stage 2(Q&A) 가중치
SPAN_CKPT = "SOP_GPT_span.pt" # Stage 4(추출형 RAG QA) 가중치

EARLY_STOP_PATIENCE = 10     # 연속 10번 평가(=200 step) 동안 개선이 없으면 중단
EARLY_STOP_MIN_DELTA = 1e-3  # 이보다 작은 감소는 "개선 없음"으로 취급

FT_STEPS = 3000
FT_LR = 3e-4                 # Stage 1보다 낮은 lr로 미세조정 (급격한 망각 방지)

SPAN_STEPS = 3000
SPAN_LR = 3e-4
SPAN_BATCH_SIZE = 32

if os.path.exists(BPE_PATH):
    vocab, merges = load_bpe(BPE_PATH)
else:
    text = load_kowikitext_data(train_chars=KOWIKI_TRAIN_CHARS) + "\n" + load_korean_chatbot_data()
    vocab, merges = train_bpe(text, BPE_VOCAB_SIZE)
    save_bpe(BPE_PATH, vocab, merges)

stoi, itos = build_vocab(vocab)
vocab_size = len(vocab)
base_set = base_alphabet(vocab)


def train_stage1(model):
    kowiki_text = load_kowikitext_data(train_chars=STAGE1_KOWIKI_TRAIN_CHARS)
    chatbot_text = load_korean_chatbot_data()

    # 업샘플링 전에 90/10으로 나눠야 train/val에 똑같은 chatbot 반복 구간이 동시에 들어가는
    # 데이터 누수(val loss가 실제보다 좋게 나오는 현상)를 피할 수 있다.
    n_kowiki_train = int(0.9 * len(kowiki_text))
    n_chat_train = int(0.9 * len(chatbot_text))
    train_text = kowiki_text[:n_kowiki_train] + "\n" + chatbot_text[:n_chat_train] * STAGE1_CHATBOT_UPSAMPLE
    val_text = kowiki_text[n_kowiki_train:] + "\n" + chatbot_text[n_chat_train:] * STAGE1_CHATBOT_UPSAMPLE
    print(f"kowiki {len(kowiki_text):,} chars, chatbot(x{STAGE1_CHATBOT_UPSAMPLE}) "
          f"{len(chatbot_text) * STAGE1_CHATBOT_UPSAMPLE:,} chars")

    train_data = torch.tensor(encode(train_text, merges, stoi, base_set), dtype=torch.long)
    val_data = torch.tensor(encode(val_text, merges, stoi, base_set), dtype=torch.long)
    print(f"train {len(train_data):,} tokens, val {len(val_data):,} tokens, vocab_size: {vocab_size}, device: {device}")

    get_batch = make_batcher(train_data, val_data)

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, device={device}")
    train_loop(model, get_batch, steps, lr, GEN_CKPT, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)

    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token
    # chat()과 동일한 생성 옵션을 줘야 "이상한 토큰 반복"이 아닌 의미 있는 샘플이 나온다.
    stop_tokens = {i for t, i in stoi.items() if t and t[-1] in ".?!"}
    print("\n--- sample ---")
    print(decode(model.generate(prompt, 200, stop_tokens=stop_tokens, temperature=0.7, top_p=0.9, repetition_penalty=1.3)[0].tolist(), itos))


def train_stage2(model):
    text = load_chatbot_qa_data()
    data = torch.tensor(encode(text, merges, stoi, base_set), dtype=torch.long)
    print(f"qa text length: {len(text):,} chars, {len(data):,} tokens, vocab_size: {vocab_size}, device: {device}")

    n_train = int(0.9 * len(data))
    get_batch = make_batcher(data[:n_train], data[n_train:])

    model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
    print(f"loaded {GEN_CKPT}, fine-tuning on {len(data):,} QA tokens, device={device}")
    train_loop(model, get_batch, FT_STEPS, FT_LR, QA_CKPT, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)


def _span_batch(examples, batch_size_n):
    """examples: (prompt_text, nfd_start, nfd_end) 리스트에서 배치 하나를 뽑는다.

    토큰 단위 연속 스트림이 아니라 example 단위로 샘플링하므로 train_utils.make_batcher는
    재사용할 수 없다 (질문마다 입력 길이가 다르고, 정답 위치라는 단일 타깃을 분류해야 함)."""
    ids_batch, start_batch, end_batch = [], [], []
    while len(ids_batch) < batch_size_n:
        prompt, nfd_start, nfd_end = random.choice(examples)
        tokens, offsets, _ = tokenize_with_offsets(prompt, merges, base_set)
        ids = [stoi[t] for t in tokens]
        if len(ids) > block_size:
            continue
        try:
            start_idx = next(i for i, (s, e) in enumerate(offsets) if e > nfd_start)
            end_idx = next(i for i, (s, e) in enumerate(offsets) if e >= nfd_end)
        except StopIteration:
            continue
        ids = ids + [0] * (block_size - len(ids))  # 우측 패딩 (causal attention이라 뒤쪽 패딩은 앞쪽에 영향 없음)
        ids_batch.append(ids)
        start_batch.append(start_idx)
        end_batch.append(end_idx)
    return (torch.tensor(ids_batch, dtype=torch.long, device=device),
            torch.tensor(start_batch, dtype=torch.long, device=device),
            torch.tensor(end_batch, dtype=torch.long, device=device))


def train_stage4(model):
    """Stage 4: KorQuAD로 추출형 QA(정답 시작/끝 토큰 위치 분류) 학습.
    생성 대신 분류라서 자기회귀 오류가 누적되지 않고, 작은 모델로도 정확한 정답 위치를 찾을 수 있다."""
    examples = rag.build_span_examples()
    random.Random(42).shuffle(examples)
    n_train = int(0.9 * len(examples))
    train_examples, val_examples = examples[:n_train], examples[n_train:]
    print(f"{len(train_examples):,} train / {len(val_examples):,} val span examples")

    model.load_body_from(torch.load(QA_CKPT, map_location=device))
    print(f"loaded body weights from {QA_CKPT}, training qa_head from scratch, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=SPAN_LR)
    best_val, best_state, no_improve = float("inf"), None, 0
    for step in range(SPAN_STEPS):
        ids, starts, ends = _span_batch(train_examples, SPAN_BATCH_SIZE)
        _, _, loss = model(ids, starts, ends)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == SPAN_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    model(*_span_batch(val_examples, SPAN_BATCH_SIZE))[2].item() for _ in range(20)
                ) / 20
            model.train()
            print(f"step {step:5d}  train {loss.item():.3f}  val {val_loss:.3f}", flush=True)
            if val_loss < best_val - EARLY_STOP_MIN_DELTA:
                best_val, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"early stopping at step {step} (best val {best_val:.3f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), SPAN_CKPT)
    print(f"saved best weights (val {best_val:.3f}) to {SPAN_CKPT}")


def _chat_span(model):
    tfidf_vectorizer, tfidf_matrix, embed_matrix, passages, norm_bounds = rag.build_index()
    chat_span(model, stoi, merges, base_set, tfidf_vectorizer, tfidf_matrix, embed_matrix, norm_bounds, passages)


SPAN_MODES = {"train_span", "chat_span"}

MODES = {
    "train": train_stage1,
    "train_qa": train_stage2,
    "train_span": train_stage4,
    "chat": lambda model: chat(model, stoi, itos, merges, base_set),
    "chat_qa": lambda model: chat_qa(model, stoi, itos, merges, base_set),
    "chat_span": _chat_span,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if mode not in MODES:
        sys.exit(f"unknown mode: {mode} (use one of {', '.join(MODES)})")

    model = (SOP_GPT_Span(vocab_size) if mode in SPAN_MODES else SOP_GPT(vocab_size)).to(device)
    MODES[mode](model)
