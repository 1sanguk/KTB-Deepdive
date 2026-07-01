"""DPO (Direct Preference Optimization) 학습 모듈.

선호(chosen) / 비선호(rejected) 응답 쌍으로 모델을 파인튜닝한다.
참고: Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290

학습 데이터 전략:
  - chosen  : 데이터셋의 정답 답변
  - rejected: 동일 배치 내 다른 질문의 답변을 섞어 부적절한 응답으로 사용
               (인퍼런스 없이 데이터셋만으로 선호 쌍을 만들 수 있어 경량)
"""

import copy
import random
import torch
import torch.nn.functional as F

from bpe import EOS_TOKEN, encode
from model import device, block_size


def _encode_qa(question: str, answer: str, stoi: dict, merges: dict, base_set: set):
    """'질문: Q\\n답변: ' 프롬프트 ids + 답변 ids + EOS id를 분리해 반환."""
    prompt_text = f"질문: {question}\n답변: "
    answer_text = f"{answer}{EOS_TOKEN}"
    prompt_ids = encode(prompt_text, merges, stoi, base_set)
    answer_ids  = encode(answer_text, merges, stoi, base_set)
    return prompt_ids, answer_ids


def _compute_log_prob(model, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
    """response_ids 각 토큰의 log-probability를 합산해 스칼라로 반환."""
    full = prompt_ids + response_ids
    if len(full) > block_size:
        full = full[-block_size:]
    input_t  = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    target_t = torch.tensor([full[1:]],  dtype=torch.long, device=device)

    logits, _ = model(input_t)
    log_probs = F.log_softmax(logits, dim=-1)   # (1, T, vocab)

    # 응답 구간만 집계 (프롬프트 토큰 제외)
    resp_start = max(0, len(prompt_ids) - 1)     # -1: next-token prediction shift
    resp_log_probs = log_probs[0, resp_start:, :] # (resp_len, vocab)
    resp_targets   = target_t[0, resp_start:]      # (resp_len,)

    if resp_targets.numel() == 0:
        return torch.tensor(0.0, device=device)
    return resp_log_probs.gather(1, resp_targets.unsqueeze(1)).squeeze(1).sum()


def _dpo_loss(
    policy_chosen:   torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen:      torch.Tensor,
    ref_rejected:    torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss: -log sigmoid(β * (π_θ(chosen) - π_θ(rejected) - π_ref(chosen) + π_ref(rejected)))."""
    ratio = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * ratio)


def _make_batch(pairs: list[tuple[str, str]], batch_size: int, stoi, merges, base_set):
    """pairs에서 batch_size개의 (prompt, chosen, rejected) 인코딩 트리플을 생성."""
    batch = random.sample(pairs, min(batch_size, len(pairs)))
    # rejected는 배치 내에서 순환 이동(circular shift)으로 만들어 다른 질문의 답변을 붙임
    shuffled = batch[1:] + batch[:1]

    result = []
    for (q, a_chosen), (_, a_rejected) in zip(batch, shuffled):
        p_ids, c_ids = _encode_qa(q, a_chosen,  stoi, merges, base_set)
        _,      r_ids = _encode_qa(q, a_rejected, stoi, merges, base_set)
        if not c_ids or not r_ids:
            continue
        result.append((p_ids, c_ids, r_ids))
    return result


def run_dpo(
    policy_model,
    pairs: list[tuple[str, str]],
    stoi: dict,
    itos: dict,
    merges: dict,
    base_set: set,
    steps: int = 1000,
    lr: float = 1e-5,
    beta: float = 0.1,
    batch_size: int = 4,
    ckpt_path: str = "SOP_GPT_dpo.pt",
    eval_every: int = 50,
):
    """DPO 학습 루프.

    policy_model : Stage 2 가중치가 로드된 SOP_GPT (학습됨)
    pairs        : (question, answer) 리스트 — chosen 기준
    """
    # 레퍼런스 모델: policy와 동일한 초기 가중치, gradient 없음
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    policy_model.train()
    opt = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    random.shuffle(pairs)
    n_train = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]

    best_val, best_state = float("inf"), None

    print(f"[dpo] {len(train_pairs):,} train / {len(val_pairs):,} val pairs, β={beta}, lr={lr}")

    for step in range(steps):
        batch = _make_batch(train_pairs, batch_size, stoi, merges, base_set)
        if not batch:
            continue

        losses = []
        for p_ids, c_ids, r_ids in batch:
            pol_c  = _compute_log_prob(policy_model, p_ids, c_ids)
            pol_r  = _compute_log_prob(policy_model, p_ids, r_ids)
            with torch.no_grad():
                ref_c = _compute_log_prob(ref_model, p_ids, c_ids)
                ref_r = _compute_log_prob(ref_model, p_ids, r_ids)
            losses.append(_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta))

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            policy_model.eval()
            val_batch = _make_batch(val_pairs, batch_size * 2, stoi, merges, base_set)
            val_losses = []
            with torch.no_grad():
                for p_ids, c_ids, r_ids in val_batch:
                    pol_c = _compute_log_prob(policy_model, p_ids, c_ids)
                    pol_r = _compute_log_prob(policy_model, p_ids, r_ids)
                    ref_c = _compute_log_prob(ref_model, p_ids, c_ids)
                    ref_r = _compute_log_prob(ref_model, p_ids, r_ids)
                    val_losses.append(_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta).item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
            policy_model.train()
            print(f"step {step:4d}  train {loss.item():.4f}  val {val_loss:.4f}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in policy_model.state_dict().items()}

    if best_state is not None:
        policy_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(policy_model.state_dict(), ckpt_path)
    print(f"[dpo] saved best weights (val {best_val:.4f}) to {ckpt_path}")
