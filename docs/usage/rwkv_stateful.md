# RWKV stateful streaming (experimental)

RWKV7 can keep its recurrent state across input chunks by using the existing
`AsyncLLM` streaming-input interface. This is an experimental inference-side
feature; it does not add automatic summarization or tool-call rewriting.

Each chunk must carry the same `session_id` and use `context_mode="stateful"`:

```python
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt

yield StreamingInput(
    prompt=TokensPrompt(prompt_token_ids=new_token_ids),
    session_id="session_xxx",
    context_mode="stateful",
)
```

The session state is kept in a GPU-resident RWKV state row. A streaming update
reuses that row and schedules only tokens after the scheduler's
`num_computed_tokens` position. When the request detaches, the row remains
reserved for the session until `delete_session` is called by the worker-side
model-state controller.

Current limitations:

- Only RWKV7 text requests are supported; multimodal and LoRA stateful chunks
  are rejected.
- Sampling parameters must remain unchanged within a session.
- State is process-local and GPU-resident. There is no CPU offload, TTL, fork,
  or cross-process recovery yet.
- `context_mode="sliding"` remains the default and keeps the existing behavior.

The reproducible probe is:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
python tools/rwkv_profile/stateful_streaming_probe.py \
  --model /path/to/rwkv7-g1h-1.5b-20260710-ctx10240.pth \
  --output results/stateful/1p5b.json
```

Use `--lengths 8192 12288 20480` to exercise the first-stage continuity
targets. The probe records time to first output and total generation time for
both sliding and stateful modes.
