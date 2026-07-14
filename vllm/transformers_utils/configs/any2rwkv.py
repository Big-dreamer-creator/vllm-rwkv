# SPDX-License-Identifier: Apache-2.0
"""Local HF config identities for Any2RWKV checkpoints."""

from transformers import PretrainedConfig


class Any2RWKVConfigBase(PretrainedConfig):
    model_type = "any2rwkv_base"
    architectures = ["Any2RWKV7ForCausalLM"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 60,
        head_dim: int = 64,
        head_size: int | None = None,
        num_heads: int | None = None,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 32,
        layer_types: list[str] | None = None,
        any2rwkv: dict | None = None,
        rope_parameters: dict | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("architectures", self.architectures)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.head_dim = int(head_dim)
        self.head_size = int(head_size or head_dim)
        self.num_heads = int(num_heads or hidden_size // head_dim)
        self.num_attention_heads = self.num_heads
        self.rms_norm_eps = float(rms_norm_eps)
        self.hidden_act = hidden_act
        self.decay_low_rank_dim = int(decay_low_rank_dim)
        self.gate_low_rank_dim = int(gate_low_rank_dim)
        self.a_low_rank_dim = int(a_low_rank_dim)
        self.v_low_rank_dim = int(v_low_rank_dim)
        self.layer_types = list(layer_types or ["rwkv7"] * num_hidden_layers)
        self.any2rwkv = dict(any2rwkv or {})
        self.rope_parameters = dict(rope_parameters or {})


class Any2RWKV7Config(Any2RWKVConfigBase):
    model_type = "any2rwkv_qwen35_rwkv7"
    architectures = ["Any2RWKV7ForCausalLM"]


class Any2RWKVProxyConfig(Any2RWKVConfigBase):
    model_type = "any2rwkv_proxy"
    architectures = ["Any2RWKVProxyForCausalLM"]


class Any2RWKVHybridConfig(Any2RWKVConfigBase):
    model_type = "any2rwkv_hybrid"
    architectures = ["Any2RWKVHybridForCausalLM"]
