import torch
import torch.nn as nn
import torch.nn.functional as F

# --- hyperparameters ---
block_size = 128     # max context length
n_embd = 256         # embedding dimension
n_head = 4           # attention heads
n_layer = 6          # transformer blocks
batch_size = 64
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
    """Multi-head scaled dot-product attention: softmax(QK^T / sqrt(d) + mask) V."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # project x to Q, K, V at once
        self.proj = nn.Linear(n_embd, n_embd)      # merge heads back together
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // n_head
        
        # each of q, k, v: (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # attention scores: how much each position attends to every earlier one
        att = q @ k.transpose(-2, -1) / head_dim**0.5        # (B, n_head, T, T)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~causal, float("-inf"))        # no peeking at the future
        att = self.drop(F.softmax(att, dim=-1))
        out = att @ v                                        # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)           # concat heads
        return self.drop(self.proj(out))

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

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SOP_GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_tokens=None, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 직전 block_size 토큰 안에서 이미 나온 토큰의 logit을 깎아 반복(루프)을 줄인다.
                for token_id in set(idx[0, -block_size:].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            if top_k is not None:
                # top_k보다 확률이 낮은 토큰은 후보에서 제외 (이상한 토큰이 뽑힐 확률을 줄임)
                kth_value = torch.topk(logits, top_k).values[:, -1:]
                logits[logits < kth_value] = float("-inf")
            if top_p is not None:
                # 누적 확률이 top_p를 넘는 지점부터의 토큰(꼬리, 즉 불필요하게 이상한 토큰)을 후보에서 제외
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p  # 첫 토큰은 항상 남김
                sorted_logits[sorted_mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break
        return idx


class SOP_GPT_Span(nn.Module):
    """추출형 QA용: SOP_GPT와 같은 transformer 본체를 쓰지만, 답을 한 토큰씩 생성하는 대신
    "참고: ..." 문맥 안에서 정답의 시작/끝 토큰 위치를 직접 분류(classification)한다.
    생성이 아니라서 자기회귀 오류가 누적되지 않고, 작은 모델로도 더 정확한 정답 위치를 찾을 수 있다."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
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
        x = self.ln_f(self.blocks(x))
        logits = self.qa_head(x)  # (B, T, 2)
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        loss = None
        if start_targets is not None:
            loss = F.cross_entropy(start_logits, start_targets) + F.cross_entropy(end_logits, end_targets)
        return start_logits, end_logits, loss