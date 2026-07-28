# SPDX-License-Identifier: Apache-2.0

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def test_rwkv_stateful_request_carries_session_metadata() -> None:
    request = ChatCompletionRequest(
        model="rwkv",
        messages=[{"role": "user", "content": "hello"}],
        rwkv_context_mode="stateful",
        rwkv_session_id="session-a",
        rwkv_session_action="continue",
    )

    assert request.rwkv_context_mode == "stateful"
    assert request.rwkv_session_id == "session-a"
    assert request.rwkv_session_action == "continue"


def test_rwkv_sliding_request_has_no_session_by_default() -> None:
    request = ChatCompletionRequest(
        model="rwkv",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.rwkv_context_mode is None
    assert request.rwkv_session_id is None
    assert request.rwkv_session_action == "create"
