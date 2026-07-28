# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.model_states.rwkv import RWKV7ModelState

pytestmark = pytest.mark.cpu_test


def _model_state(max_num_seqs: int = 4) -> RWKV7ModelState:
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=8,
        head_size=4,
        num_attention_heads=2,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        max_model_len=32768,
        dtype=torch.float16,
        get_inputs_embeds_size=lambda: 8,
    )
    scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )
    model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        tp_num_heads=2,
        wkv_state_dtype=torch.float16,
        wkv_mode="fp16",
    )
    return RWKV7ModelState(vllm_config, model, None, torch.device("cpu"))


def _request(
    req_id: str,
    session_id: str,
    token_ids: list[int],
    num_computed_tokens: int,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=token_ids,
        prefill_token_ids=token_ids.copy(),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        session_id=session_id,
        context_mode="stateful",
    )


def test_stateful_update_retains_row_and_detached_session() -> None:
    state = _model_state()
    initial = _request("req-a", "session-a", [1, 2, 3], 0)
    state.add_request(0, initial)
    row = state.req_slot_to_row[0]
    state.shift_state[:, :, row].fill_(1)
    state.wkv_state[:, row].fill_(2)
    state.elapsed[row] = 7

    update = _request("req-a", "session-a", [1, 2, 3, 4], 3)
    assert state.can_preserve_streaming_state("req-a", update)
    state.update_streaming_state(0, update)

    assert state.req_slot_to_row[0] == row
    assert torch.all(state.shift_state[:, :, row] == 1)
    assert torch.all(state.wkv_state[:, row] == 2)
    assert state.elapsed[row].item() == 7
    assert state.get_session("session-a")["processed_token_count"] == 3

    state.remove_request("req-a")
    detached = state.get_session("session-a")
    assert detached["status"] == "detached"
    assert detached["request_id"] is None
    assert row not in state.free_rows

    state.add_request(1, _request("req-a-2", "session-a", [1, 2, 3, 4], 3))
    assert state.req_slot_to_row[1] == row
    assert torch.all(state.wkv_state[:, row] == 2)


def test_stateful_sessions_are_isolated_and_lifecycle_is_explicit() -> None:
    state = _model_state()
    state.add_request(0, _request("req-a", "session-a", [1], 0))
    state.add_request(1, _request("req-b", "session-b", [2], 0))
    row_a = state.req_slot_to_row[0]
    row_b = state.req_slot_to_row[1]
    assert row_a != row_b

    state.wkv_state[:, row_a].fill_(11)
    state.wkv_state[:, row_b].fill_(22)
    with pytest.raises(RuntimeError, match="already bound"):
        state.add_request(2, _request("req-a-2", "session-a", [1], 0))

    state.remove_request("req-a")
    state.reset_session("session-a")
    assert torch.all(state.wkv_state[:, row_a] == 0)
    assert torch.all(state.wkv_state[:, row_b] == 22)

    state.delete_session("session-a")
    with pytest.raises(KeyError):
        state.get_session("session-a")
    assert state.get_session("session-b")["status"] == "active"


def test_stateful_update_rejects_backwards_progress_and_wrong_session() -> None:
    state = _model_state()
    state.add_request(0, _request("req-a", "session-a", [1, 2], 0))
    state.update_streaming_state(0, _request("req-a", "session-a", [1, 2, 3], 2))

    with pytest.raises(RuntimeError, match="moved backwards"):
        state.can_preserve_streaming_state(
            "req-a", _request("req-a", "session-a", [1], 1)
        )
    with pytest.raises(RuntimeError, match="not 'session-b'"):
        state.can_preserve_streaming_state(
            "req-a", _request("req-a", "session-b", [1, 2, 3], 2)
        )


def test_detached_sessions_are_reclaimed_after_ttl() -> None:
    state = _model_state()
    state.session_ttl_seconds = 10
    state.add_request(0, _request("req-a", "session-a", [1], 0))
    state.remove_request("req-a")
    state.sessions["session-a"].last_active_at = 100

    assert state.evict_expired_sessions(now=109) == []
    assert state.evict_expired_sessions(now=110) == ["session-a"]
    with pytest.raises(KeyError):
        state.get_session("session-a")
