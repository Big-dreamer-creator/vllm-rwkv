# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.inputs.context_window import (
    apply_context_window,
    apply_context_window_to_tokenize_params,
    resolve_context_window,
    trim_prompt_token_ids,
)
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers.rwkv_defaults import trim_rwkv_chat_messages


class _LengthTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        length = sum(len(message.get("content", "")) for message in messages)
        return list(range(length))


def test_rwkv_context_window_defaults_to_one_window() -> None:
    model_config = SimpleNamespace(
        context_window=None,
        tokenizer_mode="rwkv",
        hf_text_config=SimpleNamespace(model_type="rwkv7"),
    )

    assert resolve_context_window(model_config) == 8192


def test_explicit_or_model_context_window_wins() -> None:
    explicit = SimpleNamespace(
        context_window=16384,
        tokenizer_mode="rwkv",
        hf_text_config=SimpleNamespace(model_type="rwkv7"),
    )
    advertised = SimpleNamespace(
        context_window=None,
        tokenizer_mode="rwkv",
        hf_text_config=SimpleNamespace(model_type="rwkv7", rwkv_context_window=32768),
    )

    assert resolve_context_window(explicit) == 16384
    assert resolve_context_window(advertised) == 32768
    assert resolve_context_window(advertised, "none") is None


def test_sliding_window_keeps_latest_prompt_tokens() -> None:
    assert trim_prompt_token_ids([1, 2, 3, 4, 5], 3) == [3, 4, 5]
    assert trim_prompt_token_ids([1, 2], 3) == [1, 2]


def test_context_window_caps_renderer_input_before_validation() -> None:
    params = TokenizeParams(max_total_tokens=10240, max_output_tokens=4)

    capped = apply_context_window_to_tokenize_params(params, 8192)

    assert capped.max_total_tokens == 8196
    assert capped.max_input_tokens == 8192
    assert capped.truncate_prompt_tokens == 8192
    assert capped.truncation_side == "left"


def test_context_window_respects_output_budget() -> None:
    params = TokenizeParams(max_total_tokens=4096, max_output_tokens=1024)

    capped = apply_context_window_to_tokenize_params(params, 8192)

    assert capped.max_total_tokens == 4096
    assert capped.max_input_tokens == 3072
    assert capped.truncate_prompt_tokens == 3072


def test_none_strategy_preserves_token_and_tokenize_inputs() -> None:
    params = TokenizeParams(max_total_tokens=10240, max_output_tokens=4)
    inputs = {
        "type": "token",
        "prompt_token_ids": [1, 2, 3],
    }

    assert apply_context_window_to_tokenize_params(
        params, 2, "none"
    ) is params
    apply_context_window(inputs, 2, "none")
    assert inputs["prompt_token_ids"] == [1, 2, 3]


def test_context_window_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported context window strategy"):
        trim_prompt_token_ids([1, 2], 1, "unknown")  # type: ignore[arg-type]


def test_context_window_updates_token_aligned_metadata() -> None:
    inputs = {
        "type": "token",
        "prompt_token_ids": [1, 2, 3, 4],
        "assistant_tokens_mask": [0, 0, 1, 1],
        "prompt_token_offsets": [(0, 1), (1, 2), (2, 3), (3, 4)],
    }

    apply_context_window(inputs, 2)

    assert inputs["prompt_token_ids"] == [3, 4]
    assert inputs["assistant_tokens_mask"] == [1, 1]
    assert inputs["prompt_token_offsets"] == [(2, 3), (3, 4)]


def test_context_window_does_not_trim_multimodal_inputs() -> None:
    inputs = {
        "type": "multimodal",
        "prompt_token_ids": [1, 2, 3],
        "mm_kwargs": {},
        "mm_hashes": {},
        "mm_placeholders": {},
    }

    apply_context_window(inputs, 2)

    assert inputs["prompt_token_ids"] == [1, 2, 3]


def test_rwkv_message_window_drops_complete_old_turns() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "old result"},
        {"role": "user", "content": "new"},
    ]

    kept = trim_rwkv_chat_messages(
        messages,
        tools=None,
        tokenizer=_LengthTokenizer(),
        context_window=5,
    )

    assert [message["role"] for message in kept] == ["system", "user"]
    assert kept[-1]["content"] == "new"
