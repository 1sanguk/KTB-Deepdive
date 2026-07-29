from transformers import PretrainedConfig


class SopGptConfig(PretrainedConfig):
    """SOP_GPT 하이퍼파라미터 설정."""

    model_type = "sop_gpt"

    def __init__(
        self,
        vocab_size: int = 8003,
        block_size: int = 256,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        dropout: float = 0.2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd     = n_embd
        self.n_head     = n_head
        self.n_layer    = n_layer
        self.dropout    = dropout
        super().__init__(**kwargs)
