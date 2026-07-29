"""SOP_GPT — HuggingFace PreTrainedModel 래핑."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, QuestionAnsweringModelOutput

try:
    from .configuration_sop_gpt import SopGptConfig
except ImportError:
    from configuration_sop_gpt import SopGptConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SopGptConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv  = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        head_dim = C // self.n_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        T_kv = k.shape[2]
        att  = q @ k.transpose(-2, -1) / head_dim ** 0.5

        if T > 1:
            q_start = T_kv - T
            causal  = torch.tril(torch.ones(T_kv, T_kv, dtype=torch.bool, device=x.device))[q_start:]
            att     = att.masked_fill(~causal, float("-inf"))

        att = self.drop(F.softmax(att.float(), dim=-1).to(x.dtype))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out)), (k, v)


class Block(nn.Module):
    def __init__(self, config: SopGptConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


class SopGptPreTrainedModel(PreTrainedModel):
    config_class         = SopGptConfig
    base_model_prefix    = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class SopGptForCausalLM(SopGptPreTrainedModel, GenerationMixin):
    """SOP_GPT Decoder-only 언어 모델 (한국어 GPT-2 스타일, ~97M params)."""

    def __init__(self, config: SopGptConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding):
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        T   = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x   = self.tok_emb(input_ids) + self.pos_emb(pos)

        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv   = block(x, past_kv)
            new_kvs.append(kv)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(new_kvs) if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    # ── 커스텀 생성 메서드 (sop_llm.py 호환용) ────────────────────────────────────

    def _forward_with_cache(
        self,
        ctx: torch.Tensor,
        positions: torch.Tensor,
        past_kvs: Optional[List],
    ) -> Tuple[torch.Tensor, List]:
        x = self.tok_emb(ctx) + self.pos_emb(positions)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past_kvs[i] if past_kvs is not None else None)
            new_kvs.append(kv)
        return self.ln_f(x), new_kvs

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_ids: Optional[set],
    ) -> torch.Tensor:
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

    @torch.no_grad()
    def generate_raw(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        stop_tokens: Optional[set] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        block_size = self.config.block_size
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
            logits = self.lm_head(x)[:, -1, :].float()

            rep_ids = set(idx[0, -block_size:].tolist()) if repetition_penalty != 1.0 else None
            next_id = self._sample_logits(logits, temperature, top_k, top_p, rep_ids)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_tokens is not None and next_id.item() in stop_tokens and step >= min_new_tokens:
                break
        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        stop_tokens: Optional[set] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
    ):
        """토큰 id를 하나씩 yield하는 스트리밍 제너레이터."""
        block_size = self.config.block_size
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
            logits = self.lm_head(x)[:, -1, :].float()

            rep_ids = set(idx[0, -block_size:].tolist()) if repetition_penalty != 1.0 else None
            next_id = self._sample_logits(logits, temperature, top_k, top_p, rep_ids)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id.item()
            if stop_tokens is not None and next_id.item() in stop_tokens and step >= min_new_tokens:
                break


class SopGptForSpanExtraction(SopGptPreTrainedModel):
    """SOP_GPT 추출형 QA 모델 — 문서 안에서 정답의 시작/끝 토큰 위치를 분류."""

    def __init__(self, config: SopGptConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.qa_head = nn.Linear(config.n_embd, 2)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        T   = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x   = self.tok_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x, _ = block(x)

        x      = self.ln_f(x)
        logits = self.qa_head(x)          # (B, T, 2)
        start_logits = logits[..., 0]
        end_logits   = logits[..., 1]

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (
                F.cross_entropy(start_logits, start_positions)
                + F.cross_entropy(end_logits, end_positions)
            )

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )
