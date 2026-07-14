# SPDX-License-Identifier: Apache-2.0
"""Inference-only Any2RWKV Qwen3.5 text model."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .qwen2_moe import Qwen2MoeMLP
from .qwen3_next import Qwen3NextModel, Qwen3NextSparseMoeBlock
from .rwkv7 import EXECUTION_PROFILE, RWKV7ForCausalLM
from .utils import AutoWeightsLoader, maybe_prefix


def _linear(module: ReplicatedLinear, value: torch.Tensor) -> torch.Tensor:
    output = module(value)
    return output[0] if isinstance(output, tuple) else output


def _partial_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    theta: float,
) -> torch.Tensor:
    if rotary_dim == 0:
        return value
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=value.device,
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(value.dtype).unsqueeze(-2)
    sine = embedding.sin().to(value.dtype).unsqueeze(-2)
    rotary = value[..., :rotary_dim]
    left, right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return torch.cat(
        (rotary * cosine + rotated * sine, value[..., rotary_dim:]), dim=-1
    )


class _LowRank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        bias: bool,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.lora = nn.ModuleList(
            (
                ReplicatedLinear(
                    hidden_size,
                    rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.lora.0",
                    return_bias=False,
                ),
                nn.Identity(),
                ColumnParallelLinear(
                    rank,
                    hidden_size,
                    bias=bias,
                    gather_output=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.lora.2",
                    return_bias=False,
                ),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.project_out(self.project_in(value))

    def project_in(self, value: torch.Tensor) -> torch.Tensor:
        return _linear(self.lora[0], value)

    def project_out(self, value: torch.Tensor) -> torch.Tensor:
        return _linear(self.lora[2], value)


class Any2RWKV7Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.head_dim)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        full_heads = int(config.num_heads)
        if full_heads % self.tp_size:
            raise ValueError(
                f"Any2RWKV num_heads={full_heads} is not divisible by TP={self.tp_size}"
            )
        self.full_num_heads = full_heads
        self.num_heads = full_heads // self.tp_size
        self.local_hidden_size = self.num_heads * self.head_dim
        if self.head_dim != 64:
            raise ValueError("Any2RWKV vLLM runtime requires native head_dim=64")
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, self.hidden_size)))
        # Bare parameters use the default loader, so replicate these O(hidden)
        # vectors and select a head-aligned view at runtime.  The O(hidden^2)
        # matrices below carry the material TP memory saving.
        self.k_k = nn.Parameter(torch.empty(self.hidden_size))
        self.k_a = nn.Parameter(torch.empty(self.hidden_size))
        self.r_k = nn.Parameter(torch.empty(self.full_num_heads, self.head_dim))

        def input_projection(name: str) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.{name}_proj",
                return_bias=False,
            )

        self.r_proj = input_projection("r")
        self.k_proj = input_projection("k")
        self.v_proj = input_projection("v")
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            return_bias=False,
        )
        self.w_lora = _LowRank(
            self.hidden_size,
            int(config.decay_low_rank_dim),
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w_lora",
        )
        self.a_lora = _LowRank(
            self.hidden_size,
            int(config.a_low_rank_dim),
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.a_lora",
        )
        self.g_lora = _LowRank(
            self.hidden_size,
            int(config.gate_low_rank_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_lora",
        )
        if layer_idx:
            self.v_lora = _LowRank(
                self.hidden_size,
                int(config.v_low_rank_dim),
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.v_lora",
            )
        self.g_norm = nn.GroupNorm(
            self.full_num_heads, self.hidden_size, eps=self.head_dim * 1e-5
        )
        source_types = config.any2rwkv["source_layer_types"]
        self.source_used_rope = source_types[layer_idx] == "full_attention"
        source = config.any2rwkv["source_text_config"]
        source_head_dim = int(source.get("head_dim", self.head_dim))
        rope = config.rope_parameters
        self.rotary_dim = int(
            source_head_dim * float(rope.get("partial_rotary_factor", 1.0))
        )
        self.rotary_dim = min(self.rotary_dim, self.head_dim)
        self.rotary_dim -= self.rotary_dim % 2
        self.rope_theta = float(rope.get("rope_theta", 10_000.0))

    def forward(
        self,
        value: torch.Tensor,
        previous: torch.Tensor,
        v_first: torch.Tensor,
        state: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = value.shape[0]
        hidden = self.hidden_size
        local_hidden = self.local_hidden_size
        heads = self.num_heads
        head_dim = self.head_dim
        channel_start = self.tp_rank * local_hidden
        head_start = self.tp_rank * heads
        k_k = self.k_k.narrow(0, channel_start, local_hidden)
        k_a = self.k_a.narrow(0, channel_start, local_hidden)
        r_k = self.r_k.narrow(0, head_start, heads)
        g_norm_weight = self.g_norm.weight.narrow(
            0, channel_start, local_hidden
        )
        g_norm_bias = self.g_norm.bias.narrow(0, channel_start, local_hidden)
        delta = previous.to(value.dtype) - value
        mixed = {
            name: value + delta * getattr(self, f"x_{name}").reshape(1, hidden)
            for name in ("r", "w", "k", "v", "a", "g")
        }
        receptance = _linear(self.r_proj, mixed["r"])
        decay_logits = self.w_lora.project_out(
            torch.tanh(self.w_lora.project_in(mixed["w"]))
        )
        key = _linear(self.k_proj, mixed["k"])
        output_value = _linear(self.v_proj, mixed["v"])
        erase = torch.sigmoid(
            self.a_lora.project_out(self.a_lora.project_in(mixed["a"]))
        )
        gate = self.g_lora.project_out(
            torch.sigmoid(self.g_lora.project_in(mixed["g"]))
        )
        if self.source_used_rope:
            receptance = _partial_rope(
                receptance.view(batch, heads, head_dim),
                positions,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
            ).reshape(batch, local_hidden)
            key = _partial_rope(
                key.view(batch, heads, head_dim),
                positions,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
            ).reshape(batch, local_hidden)
        normalized_key = F.normalize(
            (key * k_k.reshape(1, local_hidden)).view(batch, heads, head_dim),
            dim=-1,
            p=2,
        ).view(batch, local_hidden)
        key = key * (1 + (erase - 1) * k_a.reshape(1, local_hidden))
        if self.layer_idx == 0:
            v_first = output_value
        else:
            value_mix = torch.sigmoid(
                self.v_lora.project_out(self.v_lora.project_in(mixed["v"]))
            )
            output_value = output_value + (v_first - output_value) * value_mix
        decay = torch.exp(-0.606531 * torch.sigmoid(decay_logits.float()))
        write = output_value.view(batch, heads, head_dim, 1) @ key.view(
            batch, heads, 1, head_dim
        )
        erase_matrix = (-normalized_key).view(batch, heads, head_dim, 1) @ (
            normalized_key * erase
        ).view(batch, heads, 1, head_dim)
        state = (
            state * decay.view(batch, heads, 1, head_dim)
            + state @ erase_matrix.float()
            + write.float()
        )
        output = (state.to(value.dtype) @ receptance.view(batch, heads, head_dim, 1)).view(
            batch, local_hidden
        )
        output = F.group_norm(
            output,
            num_groups=heads,
            weight=g_norm_weight,
            bias=g_norm_bias,
            eps=head_dim * 1e-5,
        )
        bonus = (
            receptance.view(batch, heads, head_dim)
            * key.view(batch, heads, head_dim)
            * r_k.reshape(1, heads, head_dim)
        ).sum(dim=-1, keepdim=True)
        output = output + (bonus * output_value.view(batch, heads, head_dim)).view(
            batch, local_hidden
        )
        return _linear(self.o_proj, output * gate), value, state, v_first


class Any2RWKV7Layer(nn.Module):
    def __init__(self, config, layer_idx: int, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn = Any2RWKV7Attention(
            config,
            layer_idx,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )
        source_type = config.any2rwkv["source_text_config"].get("model_type", "")
        if source_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden: torch.Tensor,
        previous: torch.Tensor,
        state: torch.Tensor,
        v_first: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden
        mixed, previous, state, v_first = self.attn(
            self.input_layernorm(hidden), previous, v_first, state, positions
        )
        hidden = residual + mixed
        return (
            hidden + self.mlp(self.post_attention_layernorm(hidden)),
            previous,
            state,
            v_first,
        )


class Any2RWKV7ForCausalLM(RWKV7ForCausalLM):
    """Qwen3.5 shell with native RWKV7 state and vLLM scheduling."""

    supports_pp = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        # The Qwen residual/shift stream is replicated, while mixer projections,
        # v_first and WKV state are sharded by complete RWKV heads.
        self.quant_config = vllm_config.quant_config
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "model.embed_tokens"),
        )
        self.model.layers = nn.ModuleList(
            Any2RWKV7Layer(
                config,
                index,
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, f"model.layers.{index}"),
            )
            for index in range(config.num_hidden_layers)
        )
        self.model.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.wkv_state_dtype = EXECUTION_PROFILE.wkv_state_dtype
        self.z = {}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def _run_tokens(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        all_hidden: bool,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        shift_state, wkv_state, elapsed = state
        if slot_indices is not None:
            indices = slot_indices.to(device=shift_state.device, dtype=torch.long)
            local_shift = shift_state.index_select(2, indices)
            local_wkv = wkv_state.index_select(1, indices)
            local_elapsed = elapsed.index_select(0, indices)
        else:
            local_shift, local_wkv, local_elapsed = shift_state, wkv_state, elapsed
        outputs: list[torch.Tensor] = []
        for token_index in range(tokens.shape[1]):
            hidden = self.model.embed_tokens(tokens[:, token_index])
            v_first = hidden.new_zeros(
                hidden.shape[0], self.tp_hidden_size
            )
            positions = local_elapsed
            for layer_index, layer in enumerate(self.model.layers):
                hidden, previous, recurrent, v_first = layer(
                    hidden,
                    local_shift[layer_index, 0],
                    local_wkv[layer_index],
                    v_first,
                    positions,
                )
                local_shift[layer_index, 0].copy_(previous.to(local_shift.dtype))
                local_wkv[layer_index].copy_(recurrent.to(local_wkv.dtype))
            local_elapsed.add_(1)
            outputs.append(self.model.norm(hidden))
        if slot_indices is not None:
            shift_state.index_copy_(2, indices, local_shift)
            wkv_state.index_copy_(1, indices, local_wkv)
            elapsed.index_copy_(0, indices, local_elapsed)
        stacked = torch.stack(outputs, dim=1)
        return stacked if all_hidden else stacked[:, -1]

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run_tokens(
            tokens, state, all_hidden=False, slot_indices=slot_indices
        )

    def forward_all_hidden(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        return self._run_tokens(tokens, state, all_hidden=True)

    def forward_varlen_hidden(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
    ) -> torch.Tensor:
        del req_id, max_t
        boundaries = query_start_loc.to(device="cpu", dtype=torch.long).tolist()
        outputs = []
        for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            row_tokens = tokens[start:end].view(1, -1)
            outputs.append(
                self._run_tokens(
                    row_tokens,
                    state,
                    all_hidden=True,
                    slot_indices=slot_indices[row : row + 1],
                ).reshape(end - start, -1)
            )
        return torch.cat(outputs, dim=0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=Qwen3NextModel.hf_to_vllm_mapper)


class Any2RWKVProxyForCausalLM(Any2RWKV7ForCausalLM):
    """Serving identity for the fully recurrent small-model pilot."""
