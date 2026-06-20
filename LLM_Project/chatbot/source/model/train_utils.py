import torch

from model import block_size, batch_size, device


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
    out = {s: sum(model(*get_batch(s))[1].item() for _ in range(20)) / 20
           for s in ("train", "val")}
    model.train()
    return out


def train_loop(model, get_batch, steps, lr, ckpt_path, patience, min_delta):
    """early stopping(patience/min_delta) 적용 학습 루프. best 가중치를 ckpt_path에 저장하고 best val loss를 반환."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for step in range(steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            est = estimate_loss(model, get_batch)
            print(f"step {step:5d}  train {est['train']:.3f}  val {est['val']:.3f}", flush=True)

            if est["val"] < best_val - min_delta:
                best_val = est["val"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"early stopping at step {step} (best val {best_val:.3f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved best weights (val {best_val:.3f}) to {ckpt_path}")
    return best_val
