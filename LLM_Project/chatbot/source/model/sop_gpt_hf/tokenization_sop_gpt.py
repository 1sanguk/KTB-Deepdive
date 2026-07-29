"""SOP_GPT 토크나이저 — 자체 구현 BPE를 HuggingFace PreTrainedTokenizer로 래핑."""

import json
import os
import unicodedata
import re
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "bpe_vocab.json"}

EOS_TOKEN       = "<|endoftext|>"
USER_TOKEN      = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SPECIAL_TOKENS  = (EOS_TOKEN, USER_TOKEN, ASSISTANT_TOKEN)
UNK             = "unk_char"  # 희귀 문자 대체 토큰

_SPECIAL_RE = re.compile(r"(<\|[^|>]+\|>)")
_WORD_RE    = re.compile(r"\S+|\s+")


def _decompose(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _compose(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _word_to_tokens(word: str, merges: dict, base_set: set) -> List[str]:
    tokens = [c if c in base_set else UNK for c in word]
    while True:
        pairs  = list(zip(tokens, tokens[1:]))
        ranked = [(merges[p], i) for i, p in enumerate(pairs) if p in merges]
        if not ranked:
            return tokens
        _, i   = min(ranked)
        tokens = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]


def _tokenize_text(text: str, merges: dict, base_set: set) -> List[str]:
    tokens = []
    for part in _SPECIAL_RE.split(text):
        if part in SPECIAL_TOKENS:
            tokens.append(part)
        elif part:
            decomposed = _decompose(part)
            for word in _WORD_RE.findall(decomposed):
                tokens.extend(_word_to_tokens(word, merges, base_set))
    return tokens


class SopGptTokenizer(PreTrainedTokenizer):
    """SOP_GPT 전용 BPE 토크나이저."""

    vocab_files_names  = VOCAB_FILES_NAMES
    model_input_names  = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        eos_token: str = EOS_TOKEN,
        bos_token: str = EOS_TOKEN,
        unk_token: str = UNK,
        pad_token: str = EOS_TOKEN,
        user_token: str = USER_TOKEN,
        assistant_token: str = ASSISTANT_TOKEN,
        **kwargs,
    ):
        with open(vocab_file, encoding="utf-8") as f:
            data = json.load(f)

        self._vocab_list  = list(data["vocab"])
        self._merges_list = data["merges"]
        self._merges      = {(a, b): rank for rank, (a, b) in enumerate(self._merges_list)}

        # 특수 토큰이 vocab에 없으면 끝에 추가
        for tok in SPECIAL_TOKENS:
            if tok not in self._vocab_list:
                self._vocab_list.append(tok)

        self._stoi    = {t: i for i, t in enumerate(self._vocab_list)}
        self._itos    = {i: t for i, t in enumerate(self._vocab_list)}
        self._base_set = {t for t in self._vocab_list if len(t) == 1}

        super().__init__(
            eos_token=eos_token,
            bos_token=bos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=[user_token, assistant_token],
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_list)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._stoi)

    def _tokenize(self, text: str) -> List[str]:
        return _tokenize_text(text, self._merges, self._base_set)

    def _convert_token_to_id(self, token: str) -> int:
        return self._stoi.get(token, self._stoi.get(UNK, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._itos.get(index, UNK)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        filtered = [t for t in tokens if t not in SPECIAL_TOKENS]
        return _compose("".join(filtered))

    def tokenize_with_offsets(self, text: str):
        """각 토큰의 decompose(text) 내 [start, end) 위치도 반환 (SopGptForSpanExtraction 전용)."""
        decomposed = _decompose(text)
        tokens, offsets = [], []
        pos = 0
        for word in _WORD_RE.findall(decomposed):
            for t in _word_to_tokens(word, self._merges, self._base_set):
                tokens.append(t)
                offsets.append((pos, pos + len(t)))
                pos += len(t)
        return tokens, offsets, decomposed

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        fname    = (filename_prefix + "-" if filename_prefix else "") + "bpe_vocab.json"
        out_path = os.path.join(save_directory, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"vocab": self._vocab_list, "merges": self._merges_list},
                f,
                ensure_ascii=False,
                indent=2,
            )
        return (out_path,)
