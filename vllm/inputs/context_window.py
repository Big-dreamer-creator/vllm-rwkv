# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-side context window policies.

The policy is intentionally applied after rendering and tokenization.  This
keeps it independent of a particular chat template and makes it useful for
future message-aware, chunked, or model-state-aware policies.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal

from vllm.inputs.engine import DecoderEngineInput

ContextWindowStrategy = Literal["none", "sliding_window"]
RWKV_DEFAULT_CONTEXT_WINDOW = 8192


def resolve_context_window(
    model_config: Any,
    strategy: ContextWindowStrategy = "sliding_window",
) -> int | None:
    """Resolve an effective window without changing ``max_model_len``.

    Current RWKV checkpoints default to one 8192-token window. A future
    checkpoint can advertise a larger window through ``context_window`` or
    ``rwkv_context_window``, and an explicit engine argument always wins.
    """
    if strategy == "none":
        return None

    configured_window = getattr(model_config, "context_window", None)
    if configured_window is not None:
        return configured_window

    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    tokenizer_mode = getattr(model_config, "tokenizer_mode", None)
    if model_type != "rwkv7" and tokenizer_mode != "rwkv":
        return None

    model_window = getattr(hf_config, "context_window", None)
    if model_window is None:
        model_window = getattr(hf_config, "rwkv_context_window", None)
    return model_window or RWKV_DEFAULT_CONTEXT_WINDOW


def trim_prompt_token_ids(
    prompt_token_ids: Sequence[int],
    context_window: int | None,
    strategy: ContextWindowStrategy = "sliding_window",
) -> list[int]:
    """Trim prompt token IDs according to an inference context policy.

    ``context_window`` limits the prompt carried into a new inference request;
    it does not change the model's configured maximum context length.  The
    latter remains available for future long-context checkpoints.
    """
    if context_window is None or strategy == "none":
        return list(prompt_token_ids)
    if strategy != "sliding_window":
        raise ValueError(f"Unsupported context window strategy: {strategy!r}")
    if context_window < 1:
        raise ValueError(f"context_window must be positive, got {context_window!r}")
    if len(prompt_token_ids) <= context_window:
        return list(prompt_token_ids)
    return list(prompt_token_ids[-context_window:])


def apply_context_window_to_tokenize_params(
    tokenize_params: Any,
    context_window: int | None,
    strategy: ContextWindowStrategy = "sliding_window",
) -> Any:
    """Apply the prompt window before renderer-side length validation.

    The renderer validates tokenized prompts before the engine input
    processor runs, so a window policy must also enable left truncation at
    this earlier stage.
    """
    if context_window is None or strategy == "none":
        return tokenize_params
    if strategy != "sliding_window":
        raise ValueError(f"Unsupported context window strategy: {strategy!r}")
    if context_window < 1:
        raise ValueError(f"context_window must be positive, got {context_window!r}")

    max_output_tokens = tokenize_params.max_output_tokens or 0
    max_total_tokens = tokenize_params.max_total_tokens
    if max_total_tokens is not None:
        context_window = min(context_window, max_total_tokens - max_output_tokens)
    if context_window < 1:
        return tokenize_params

    truncate_prompt_tokens = tokenize_params.truncate_prompt_tokens
    if truncate_prompt_tokens is None or truncate_prompt_tokens < 0:
        truncate_prompt_tokens = context_window
    else:
        truncate_prompt_tokens = min(truncate_prompt_tokens, context_window)

    return replace(
        tokenize_params,
        max_total_tokens=context_window + max_output_tokens,
        truncate_prompt_tokens=truncate_prompt_tokens,
        truncation_side=tokenize_params.truncation_side or "left",
    )


def apply_context_window(
    decoder_inputs: DecoderEngineInput,
    context_window: int | None,
    strategy: ContextWindowStrategy = "sliding_window",
) -> DecoderEngineInput:
    """Apply the configured prompt window to a decoder engine input.

    Multimodal and embedding inputs are left unchanged until their metadata
    can be truncated atomically.  RWKV text inputs use the token path.
    """
    if context_window is None or strategy == "none":
        return decoder_inputs
    if decoder_inputs["type"] != "token":
        return decoder_inputs

    prompt_token_ids = decoder_inputs["prompt_token_ids"]
    if len(prompt_token_ids) <= context_window:
        return decoder_inputs

    kept = trim_prompt_token_ids(prompt_token_ids, context_window, strategy)
    decoder_inputs["prompt_token_ids"] = kept

    for field_name in ("assistant_tokens_mask", "prompt_token_offsets"):
        field = decoder_inputs.get(field_name)  # type: ignore[call-overload]
        if field is not None:
            decoder_inputs[field_name] = field[-len(kept) :]  # type: ignore[literal-required]

    return decoder_inputs
