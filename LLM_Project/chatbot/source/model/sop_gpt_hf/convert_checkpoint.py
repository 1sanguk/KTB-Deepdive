"""SOP_GPT .pt 체크포인트를 HuggingFace 형식으로 변환하는 스크립트.

사용법:
    python convert_checkpoint.py

출력:
    source/model/sop_gpt_hf/output/       ← SopGptForCausalLM (gen)
    source/model/sop_gpt_hf/output_qa/    ← SopGptForCausalLM (qa)
    source/model/sop_gpt_hf/output_span/  ← SopGptForSpanExtraction (span)
"""

import json
import shutil
import sys
from pathlib import Path

import torch

HERE            = Path(__file__).resolve().parent
MODEL_DIR       = HERE.parent / "sop_model"
OUTPUT_DIR      = HERE / "output"
OUTPUT_QA_DIR   = HERE / "output_qa"
OUTPUT_SPAN_DIR = HERE / "output_span"

sys.path.insert(0, str(HERE))

from configuration_sop_gpt import SopGptConfig          # noqa: E402
from modeling_sop_gpt import SopGptForCausalLM, SopGptForSpanExtraction  # noqa: E402
from tokenization_sop_gpt import SopGptTokenizer         # noqa: E402


_CODE_FILES = (
    "configuration_sop_gpt.py",
    "modeling_sop_gpt.py",
    "tokenization_sop_gpt.py",
)


def _remap_lm_keys(state_dict: dict) -> dict:
    """head.weight → lm_head.weight (LM 모델용)."""
    return {
        ("lm_head.weight" if k == "head.weight" else k): v
        for k, v in state_dict.items()
    }


def _load_state_dict(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu")
    return raw if isinstance(raw, dict) and not ("model_state_dict" in raw) else raw.get("model_state_dict", raw)


def _save_hf(model, config, tokenizer, output_dir: Path, auto_model_key: str, auto_model_cls: str):
    """모델·config·토크나이저 저장 + 코드 파일 복사 + auto_map 추가."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    for fname in _CODE_FILES:
        shutil.copy(HERE / fname, output_dir / fname)

    cfg_path = output_dir / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    cfg_data["auto_map"] = {
        "AutoConfig":    "configuration_sop_gpt.SopGptConfig",
        auto_model_key:  f"modeling_sop_gpt.{auto_model_cls}",
        "AutoTokenizer": "tokenization_sop_gpt.SopGptTokenizer",
    }
    cfg_path.write_text(json.dumps(cfg_data, indent=2, ensure_ascii=False))
    print(f"  저장 완료: {output_dir}")


def _make_config() -> SopGptConfig:
    return SopGptConfig(
        vocab_size=8003,
        block_size=256,
        n_embd=768,
        n_head=12,
        n_layer=12,
        dropout=0.2,
        bos_token_id=8000,
        eos_token_id=8000,
    )


def convert_gen(config: SopGptConfig, tokenizer: SopGptTokenizer):
    print("\n[gen] SOP_GPT.pt → output/")
    state_dict = _remap_lm_keys(_load_state_dict(MODEL_DIR / "SOP_GPT.pt"))
    model = SopGptForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ missing: {missing}")
    if unexpected:
        print(f"  ⚠ unexpected: {unexpected}")
    _save_hf(model, config, tokenizer, OUTPUT_DIR, "AutoModelForCausalLM", "SopGptForCausalLM")

    print("  [검증] from_pretrained 로드 테스트")
    m2 = SopGptForCausalLM.from_pretrained(OUTPUT_DIR)
    t2 = SopGptTokenizer.from_pretrained(OUTPUT_DIR)
    ids = t2.encode("<|user|>안녕<|assistant|>", return_tensors="pt")
    with torch.no_grad():
        out = m2(input_ids=ids)
    print(f"  logits: {out.logits.shape}")


def convert_qa(config: SopGptConfig, tokenizer: SopGptTokenizer):
    print("\n[qa]  SOP_GPT_qa.pt → output_qa/")
    state_dict = _remap_lm_keys(_load_state_dict(MODEL_DIR / "SOP_GPT_qa.pt"))
    model = SopGptForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ missing: {missing}")
    if unexpected:
        print(f"  ⚠ unexpected: {unexpected}")
    _save_hf(model, config, tokenizer, OUTPUT_QA_DIR, "AutoModelForCausalLM", "SopGptForCausalLM")


def convert_span(config: SopGptConfig, tokenizer: SopGptTokenizer):
    print("\n[span] SOP_GPT_span.pt → output_span/")
    state_dict = _load_state_dict(MODEL_DIR / "SOP_GPT_span.pt")
    model = SopGptForSpanExtraction(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ missing: {missing}")
    if unexpected:
        print(f"  ⚠ unexpected: {unexpected}")
    _save_hf(model, config, tokenizer, OUTPUT_SPAN_DIR, "AutoModelForQuestionAnswering", "SopGptForSpanExtraction")


def main():
    config    = _make_config()
    tokenizer = SopGptTokenizer(vocab_file=str(MODEL_DIR / "bpe_vocab.json"))

    convert_gen(config, tokenizer)
    convert_qa(config, tokenizer)
    convert_span(config, tokenizer)

    print("\n변환 완료!")
    print(f"  gen  → {OUTPUT_DIR}")
    print(f"  qa   → {OUTPUT_QA_DIR}")
    print(f"  span → {OUTPUT_SPAN_DIR}")


if __name__ == "__main__":
    main()
