import math
import torch

from model import block_size, batch_size, device

# bf16 autocast: CUDA와 MPS(Apple Silicon) 모두 지원
_AMP_ENABLED = device in ("cuda", "mps")
_AMP_DTYPE = torch.bfloat16

if _AMP_ENABLED:
    print(f"[train_utils] bf16 autocast 활성화 (device={device})")


def _autocast():
    return torch.autocast(device_type=device, dtype=_AMP_DTYPE, enabled=_AMP_ENABLED)


def make_batcher(train_data, val_data):
    """train_data/val_data(1D LongTensor)에서 (x, y) 배치를 뽑는 get_batch 함수를 만든다."""
    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch):
    """Average loss over several batches, with dropout off."""
    model.eval()
    out = {}
    for s in ("train", "val"):
        total = 0
        for _ in range(20):
            x, y = get_batch(s)
            with _autocast():
                _, loss = model(x, y)
            total += loss.item()
        out[s] = total / 20
    model.train()
    return out


def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup → cosine decay 학습률 스케줄.

    초반 warmup_steps 동안 0 → max_lr 선형 증가,
    이후 cosine 곡선으로 min_lr까지 감소한다.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def train_loop(model, get_batch, steps, lr, ckpt_path, patience, min_delta,
               max_grad_norm=1.0, warmup_steps=200, min_lr_ratio=0.1, accum_steps=4):
    """early stopping(patience/min_delta) 적용 학습 루프. best 가중치를 ckpt_path에 저장하고 best val loss를 반환.

    warmup_steps : linear warmup 구간 (step 수)
    min_lr_ratio : cosine decay 최소 lr = lr * min_lr_ratio
    accum_steps  : gradient accumulation 스텝 수 (유효 배치 = batch_size * accum_steps)
    """
    min_lr = lr * min_lr_ratio
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for step in range(steps):
        # 학습률 갱신 (warmup → cosine)
        cur_lr = get_lr(step, lr, min_lr, warmup_steps, steps)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # gradient accumulation + bf16 autocast
        opt.zero_grad()
        for _ in range(accum_steps):
            x, y = get_batch()
            with _autocast():
                _, loss = model(x, y)
            (loss / accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            est = estimate_loss(model, get_batch)
            print(f"step {step:5d}  lr {cur_lr:.2e}  train {est['train']:.3f}  val {est['val']:.3f}", flush=True)

            if est["val"] < best_val - min_delta:
                best_val = est["val"]
                # best_state를 CPU로 옮겨 GPU/MPS 메모리 절감
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"early stopping at step {step} (best val {best_val:.3f})")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved best weights (val {best_val:.3f}) to {ckpt_path}")
    return best_val
