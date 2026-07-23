import builtins
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import torch

_orig_print = builtins.print
def _tprint(*args, **kwargs):
    ts = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    _orig_print(f"[{ts}]", *args, **kwargs)
builtins.print = _tprint

from chat import chat, chat_qa, chat_span
from model import device, steps, lr, batch_size, SOP_GPT, SOP_GPT_Span, block_size
from tokenizer import (load_korean_chatbot_data, load_kowikitext_data,
                        load_chatbot_qa_data, load_chatbot_qa_pairs, load_aihub_conversation_data)
from bpe import train_bpe, build_vocab, base_alphabet, encode, decode, save_bpe, load_bpe, tokenize_with_offsets, EOS_TOKEN
from train_utils import make_batcher, train_loop, get_lr

# rag/ 패키지와 langchain/ 디렉토리(서빙용 검색기)를 모듈 검색 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import rag
from lc.retriever import build_hybrid_retriever

BPE_VOCAB_SIZE = 8000
BPE_PATH = "bpe_vocab.json"
KOWIKI_TRAIN_CHARS = 80_000_000  # kowikitext train split은 ~1.6GB라 앞부분만 사용 (BPE 학습용 전체 코퍼스 기준)

# Stage1 학습 코퍼스 믹스: kowikitext(위키체)가 chatbot 데이터(구어체)보다 250배 가까이 많아서
# "오늘 날씨는" 같은 구어 이어쓰기에도 위키 말투가 나오는 문제가 있었음 -> kowiki 비중을 줄이고
# chatbot 데이터를 업샘플링(반복)해서 두 말투의 비중을 좀 더 비슷하게 맞춤
STAGE1_KOWIKI_TRAIN_CHARS = 15_000_000
STAGE1_AIHUB_MAX_CHARS   = 50_000_000  # 전체 33억자 중 5000만자만 사용 (OOM 방지)
STAGE1_CHATBOT_UPSAMPLE = 15

GEN_CKPT = "SOP_GPT.pt"       # Stage 1(이어쓰기) 가중치
QA_CKPT = "SOP_GPT_qa.pt"     # Stage 2(Q&A) 가중치
SPAN_CKPT = "SOP_GPT_span.pt" # Stage 4(추출형 RAG QA) 가중치

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-3

FT_STEPS = 3000
FT_LR = 3e-4
FT_ACCUM_STEPS = 8           # batch_size=8 × accum=8 → 유효 배치 64

SPAN_STEPS = 6000
SPAN_LR = 3e-4
SPAN_BATCH_SIZE = 8

DPO_CKPT = "SOP_GPT_dpo.pt"
DPO_STEPS = 1000
DPO_LR = 1e-5                # SFT보다 훨씬 작은 lr (가중치 급변 방지)
DPO_BETA = 0.1               # KL 페널티 강도 (0.1 = 표준값)

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
    aihub_text = load_aihub_conversation_data(max_chars=STAGE1_AIHUB_MAX_CHARS)

    # 업샘플링 전에 90/10으로 나눠야 train/val에 똑같은 chatbot 반복 구간이 동시에 들어가는
    # 데이터 누수(val loss가 실제보다 좋게 나오는 현상)를 피할 수 있다.
    n_kowiki_train = int(0.9 * len(kowiki_text))
    n_chat_train = int(0.9 * len(chatbot_text))
    n_aihub_train = int(0.9 * len(aihub_text))
    print(f"kowiki {len(kowiki_text[:n_kowiki_train]):,} chars, chatbot(x{STAGE1_CHATBOT_UPSAMPLE}) "
          f"{len(chatbot_text[:n_chat_train]) * STAGE1_CHATBOT_UPSAMPLE:,} chars, "
          f"aihub {len(aihub_text[:n_aihub_train]):,} chars")

    # 문자열 *(15)를 피하고 텐서 repeat으로 업샘플링:
    # 기존 방식은 "kowiki + chatbot*15" 거대 문자열 → Python 리스트 → 텐서 순으로 RAM에 세 벌이 뜬다.
    # 아래 방식은 각각 인코딩 후 즉시 문자열을 해제하고, repeat은 텐서 수준에서만 수행한다.
    train_kowiki = torch.tensor(encode(kowiki_text[:n_kowiki_train], merges, stoi, base_set), dtype=torch.long)
    val_kowiki   = torch.tensor(encode(kowiki_text[n_kowiki_train:],  merges, stoi, base_set), dtype=torch.long)
    del kowiki_text; gc.collect()

    train_chatbot = torch.tensor(encode(chatbot_text[:n_chat_train], merges, stoi, base_set), dtype=torch.long)
    val_chatbot   = torch.tensor(encode(chatbot_text[n_chat_train:],  merges, stoi, base_set), dtype=torch.long)
    del chatbot_text; gc.collect()

    train_aihub = torch.tensor(encode(aihub_text[:n_aihub_train], merges, stoi, base_set), dtype=torch.long)
    val_aihub   = torch.tensor(encode(aihub_text[n_aihub_train:],  merges, stoi, base_set), dtype=torch.long)
    del aihub_text; gc.collect()

    train_data = torch.cat([train_kowiki, train_chatbot.repeat(STAGE1_CHATBOT_UPSAMPLE), train_aihub])
    val_data   = torch.cat([val_kowiki,   val_chatbot.repeat(STAGE1_CHATBOT_UPSAMPLE),   val_aihub])
    del train_kowiki, val_kowiki, train_chatbot, val_chatbot, train_aihub, val_aihub; gc.collect()

    print(f"train {len(train_data):,} tokens, val {len(val_data):,} tokens, vocab_size: {vocab_size}, device: {device}")

    get_batch = make_batcher(train_data, val_data)

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, device={device}")
    train_loop(model, get_batch, steps, lr, GEN_CKPT, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)

    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    eos_id = stoi.get(EOS_TOKEN)
    stop_tokens = {i for t, i in stoi.items() if t and t[-1] in ".?!"}
    if eos_id is not None:
        stop_tokens.add(eos_id)
    print("\n--- sample ---")
    print(decode(model.generate(prompt, 200, stop_tokens=stop_tokens, temperature=0.7, top_p=0.9, repetition_penalty=1.3)[0].tolist(), itos))


def train_stage2(model):
    # chatbot + KorQuAD 모두 EOS 포함 포맷으로 로드
    chatbot_text = load_chatbot_qa_data()
    korquad_text = rag.load_korquad_qa_data()
    text = chatbot_text + korquad_text
    data = torch.tensor(encode(text, merges, stoi, base_set), dtype=torch.long)
    print(f"[stage2] chatbot {len(chatbot_text):,} + korquad {len(korquad_text):,} chars"
          f" → {len(data):,} tokens (EOS 포함), vocab_size={vocab_size}, device={device}")

    n_train = int(0.9 * len(data))
    get_batch = make_batcher(data[:n_train], data[n_train:])

    model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
    print(f"loaded {GEN_CKPT}, fine-tuning on {len(data):,} QA tokens")
    train_loop(model, get_batch, FT_STEPS, FT_LR, QA_CKPT,
               EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, accum_steps=FT_ACCUM_STEPS)


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
    gc.collect()
    torch.mps.empty_cache()
    examples = rag.build_span_examples()
    random.Random(42).shuffle(examples)
    n_train = int(0.9 * len(examples))
    train_examples, val_examples = examples[:n_train], examples[n_train:]
    print(f"{len(train_examples):,} train / {len(val_examples):,} val span examples")

    model.load_body_from(torch.load(QA_CKPT, map_location=device))
    print(f"loaded body weights from {QA_CKPT}, training qa_head from scratch, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=SPAN_LR)
    min_lr = SPAN_LR * 0.1
    best_val, best_state, no_improve = float("inf"), None, 0
    for step in range(SPAN_STEPS):
        cur_lr = get_lr(step, SPAN_LR, min_lr, warmup_steps=100, total_steps=SPAN_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        ids, starts, ends = _span_batch(train_examples, SPAN_BATCH_SIZE)
        _, _, loss = model(ids, starts, ends)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == SPAN_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    model(*_span_batch(val_examples, SPAN_BATCH_SIZE))[2].item() for _ in range(5)
                ) / 5
            model.train()
            print(f"step {step:5d}  lr {cur_lr:.2e}  train {loss.item():.3f}  val {val_loss:.3f}", flush=True)
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


def train_dpo(model):
    """Stage 5: DPO로 선호 응답 학습. Stage 2 가중치에서 시작한다.

    사전에 scripts/make_dpo_data.py를 실행해 dpo_data.json을 생성해야 한다.
    """
    # 지연 로딩: dpo.py가 torch DPO 루프를 포함하므로 train_dpo() 호출 시에만 로드.
    from dpo import run_dpo

    dpo_data_path = Path(__file__).resolve().parent / "dpo_data.json"
    if not dpo_data_path.exists():
        print(f"[dpo] ERROR: {dpo_data_path} 없음.")
        print("[dpo] 먼저 'cd source && python ../scripts/make_dpo_data.py' 를 실행하세요.")
        return

    triples = json.loads(dpo_data_path.read_text(encoding="utf-8"))
    print(f"[dpo] {len(triples):,}쌍 로드 완료: {dpo_data_path}")

    model.load_state_dict(torch.load(QA_CKPT, map_location=device))
    print(f"[dpo] loaded {QA_CKPT} as policy init")

    run_dpo(model, triples, stoi, itos, merges, base_set,
            steps=DPO_STEPS, lr=DPO_LR, beta=DPO_BETA, ckpt_path=DPO_CKPT)


def _chat_span(model):
    hybrid_retriever = build_hybrid_retriever()
    chat_span(model, stoi, merges, base_set, hybrid_retriever)


SPAN_MODES = {"train_span", "chat_span"}

MODES = {
    "train":      train_stage1,
    "train_qa":   train_stage2,
    "train_span": train_stage4,
    "train_dpo":  train_dpo,
    "chat":       lambda model: chat(model, stoi, itos, merges, base_set),
    "chat_qa":    lambda model: chat_qa(model, stoi, itos, merges, base_set),
    "chat_span":  _chat_span,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if mode not in MODES:
        sys.exit(f"unknown mode: {mode} (use one of {', '.join(MODES)})")

    model = (SOP_GPT_Span(vocab_size) if mode in SPAN_MODES else SOP_GPT(vocab_size)).to(device)
    MODES[mode](model)
