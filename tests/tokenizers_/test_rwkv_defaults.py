# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.rwkv_defaults import render_rwkv_chat_template

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {"type": "object"},
        },
    }
]


def test_rwkv_tool_template_defaults_to_compact_json() -> None:
    rendered = render_rwkv_chat_template(
        [{"role": "user", "content": "Weather in Paris?"}],
        TOOLS,
        add_generation_prompt=True,
    )

    assert '{"name":"tool_name","arguments":{"key":"value"}}' in rendered
    assert "```json" not in rendered
    assert "Do not include tool call IDs" in rendered


def test_rwkv_tool_template_can_use_legacy_markdown() -> None:
    rendered = render_rwkv_chat_template(
        [{"role": "user", "content": "Weather in Paris?"}],
        TOOLS,
        add_generation_prompt=True,
        rwkv_tool_call_format="legacy_markdown",
    )

    assert "**Tool Call:**" in rendered
    assert "```json" in rendered
