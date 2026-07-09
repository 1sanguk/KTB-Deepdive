import torch
import torch.nn as nn
import torch.nn.functional as F

# --- hyperparameters ---
block_size = 256     # max context length
n_embd = 768         # embedding dimension (512→768, ~97M params)
n_head = 12          # attention heads (head_dim=64 유지)
n_layer = 12         # transformer blocks
batch_size = 8    # 768-dim은 메모리를 더 씀 — accum_steps=8로 유효 배치 64 유지
steps = 20000
lr = 1e-3
dropout = 0.2
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(1337)


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with optional KV cache.

    past_kv: (past_k, past_v) tensors from the previous step, or None.
    Returns (output, (k_full, v_full)) so callers can cache and reuse K/V.
    """

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        head_dim = C // n_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        T_kv = k.shape[2]
        att = q @ k.transpose(-2, -1) / head_dim ** 0.5  # (B, n_head, T, T_kv)

        if T > 1:
            # q position i (global) = T_kv - T + i, so it may attend to k positions 0..T_kv-T+i
            q_start = T_kv - T
            causal = torch.tril(torch.ones(T_kv, T_kv, dtype=torch.bool, device=x.device))[q_start:]
            att = att.masked_fill(~causal, float("-inf"))
        # T == 1: single new token may attend to all cached keys — no mask needed

        # softmax in fp32 for numerical stability, then cast back
        att = self.drop(F.softmax(att.float(), dim=-1).to(x.dtype))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out)), (k, v)


class Block(nn.Module):
    """Transformer decoder block: causal self-attention + feed-forward."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


def _sample_logits(logits, temperature, top_k, top_p, repetition_ids):
    """공통 샘플링 로직. logits: (1, vocab) float32."""
    logits = logits / temperature
    if repetition_ids is not None:
        for tid in repetition_ids:
            logits[0, tid] = logits[0, tid] / 1.3 if logits[0, tid] > 0 else logits[0, tid] * 1.3
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[:, -1:]
        logits[logits < kth] = float("-inf")
    if top_p is not None:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
        sl[cp - F.softmax(sl, dim=-1) > top_p] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


class SOP_GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x, _ = block(x)  # past_kv=None during training
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def _forward_with_cache(self, ctx, positions, past_kvs):
        """KV cache를 사용하는 단일 forward pass. (x, new_kvs) 반환."""
        x = self.tok_emb(ctx) + self.pos_emb(positions)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past_kvs[i] if past_kvs is not None else None)
            new_kvs.append(kv)
        return self.ln_f(x), new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_tokens=None, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0, min_new_tokens=0):
        past_kvs = None
        for step in range(max_new_tokens):
            if past_kvs is None:
                ctx = idx[:, -block_size:]
                positions = torch.arange(ctx.shape[1], device=idx.device)
            else:
                cache_len = past_kvs[0][0].shape[2]
                if cache_len >= block_size:
                    # 캐시가 꽉 찼으면 리셋 후 전체 재계산
                    past_kvs = None
                    ctx = idx[:, -block_size:]
                    positions = torch.arange(ctx.shape[1], device=idx.device)
                else:
                    ctx = idx[:, -1:]
                    positions = torch.tensor([cache_len], device=idx.device)

            x, past_kvs = self._forward_with_cache(ctx, positions, past_kvs)
            logits = self.head(x)[:, -1, :].float()

            rep_ids = set(idx[0, -block_size:].tolist()) if repetition_penalty != 1.0 else None
            next_id = _sample_logits(logits, temperature, top_k, top_p, rep_ids)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_tokens is not None and next_id.item() in stop_tokens and step >= min_new_tokens:
                break
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, stop_tokens=None, temperature=1.0,
                        top_k=None, top_p=None, repetition_penalty=1.0, min_new_tokens=0):
        """토큰을 하나씩 yield하는 스트리밍 버전 (KV cache 적용)."""
        past_kvs = None
        for step in range(max_new_tokens):
            if past_kvs is None:
                ctx = idx[:, -block_size:]
                positions = torch.arange(ctx.shape[1], device=idx.device)
            else:
                cache_len = past_kvs[0][0].shape[2]
                if cache_len >= block_size:
                    past_kvs = None
                    ctx = idx[:, -block_size:]
                    positions = torch.arange(ctx.shape[1], device=idx.device)
                else:
                    ctx = idx[:, -1:]
                    positions = torch.tensor([cache_len], device=idx.device)

            x, past_kvs = self._forward_with_cache(ctx, positions, past_kvs)
            logits = self.head(x)[:, -1, :].float()

            rep_ids = set(idx[0, -block_size:].tolist()) if repetition_penalty != 1.0 else None
            next_id = _sample_logits(logits, temperature, top_k, top_p, rep_ids)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id.item()
            if stop_tokens is not None and next_id.item() in stop_tokens and step >= min_new_tokens:
                break


class SOP_GPT_Span(nn.Module):
    """추출형 QA용: SOP_GPT와 같은 transformer 본체를 쓰지만, 답을 한 토큰씩 생성하는 대신
    "참고: ..." 문맥 안에서 정답의 시작/끝 토큰 위치를 직접 분류(classification)한다.
    생성이 아니라서 자기회귀 오류가 누적되지 않고, 작은 모델로도 더 정확한 정답 위치를 찾을 수 있다."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.qa_head = nn.Linear(n_embd, 2)  # 위치별 (start_logit, end_logit)

    def load_body_from(self, state_dict):
        """SOP_GPT 체크포인트의 tok_emb/pos_emb/blocks/ln_f만 가져와 초기화한다 (qa_head는 새로 학습)."""
        own_state = self.state_dict()
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
        self.load_state_dict(own_state)

    def forward(self, idx, start_targets=None, end_targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x, _ = block(x)  # KV cache 불필요 (분류 태스크)
        x = self.ln_f(x)
        logits = self.qa_head(x)  # (B, T, 2)
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        loss = None
        if start_targets is not None:
            loss = F.cross_entropy(start_logits, start_targets) + F.cross_entropy(end_logits, end_targets)
        return start_logits, end_logits, loss
