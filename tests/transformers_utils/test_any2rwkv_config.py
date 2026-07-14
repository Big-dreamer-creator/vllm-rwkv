# SPDX-License-Identifier: Apache-2.0

import json

from vllm.transformers_utils.config import get_config


def _write_config(tmp_path, *, model_type, architecture, layers):
    source_types = ["linear_attention", "full_attention"] * ((layers + 1) // 2)
    payload = {
        "model_type": model_type,
        "architectures": [architecture],
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": layers,
        "head_dim": 64,
        "head_size": 64,
        "num_heads": 1,
        "rms_norm_eps": 1e-6,
        "layer_types": ["rwkv7"] * layers,
        "any2rwkv": {
            "source_layer_types": source_types[:layers],
            "source_text_config": {"model_type": "qwen3_5_text", "head_dim": 16},
        },
        "rope_parameters": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
    }
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_final_any2rwkv_config_loads_without_remote_code(tmp_path):
    _write_config(
        tmp_path,
        model_type="any2rwkv_qwen35_rwkv7",
        architecture="Any2RWKV7ForCausalLM",
        layers=60,
    )
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.model_type == "any2rwkv_qwen35_rwkv7"
    assert config.architectures == ["Any2RWKV7ForCausalLM"]
    assert config.layer_types == ["rwkv7"] * 60


def test_proxy_identity_remains_distinct_and_loadable(tmp_path):
    _write_config(
        tmp_path,
        model_type="any2rwkv_proxy",
        architecture="Any2RWKVProxyForCausalLM",
        layers=24,
    )
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.model_type == "any2rwkv_proxy"
    assert config.architectures == ["Any2RWKVProxyForCausalLM"]
    assert config.num_hidden_layers == 24
