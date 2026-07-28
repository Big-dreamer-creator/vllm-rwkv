# RWKV stateful streaming (experimental)

RWKV7 can keep its recurrent state across input chunks by using the existing
`AsyncLLM` streaming-input interface. The OpenAI Chat Completions endpoint can
use the same mechanism with `rwkv_context_mode="stateful"` and an explicit
`rwkv_session_id`.

Stateful Chat Completions are incremental: the first request uses
`rwkv_session_action="create"` and later requests send only new messages with
`rwkv_session_action="continue"`.

```json
{
  "model": "rwkv7-g1h-1.5b",
  "messages": [{"role": "user", "content": "Continue the story."}],
  "max_completion_tokens": 128,
  "rwkv_context_mode": "stateful",
  "rwkv_session_id": "session_xxx",
  "rwkv_session_action": "create"
}
```

The response remains a normal OpenAI Chat Completion. Tool calls use the
configured RWKV tool parser and are returned in `message.tool_calls`; a later
request can continue the same session with the assistant tool-call message and
the tool result.

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
reserved for the session until it is reset, deleted, or reclaimed by TTL.

The HTTP lifecycle endpoints are:

```text
GET    /v1/rwkv/sessions/{session_id}
POST   /v1/rwkv/sessions/{session_id}/reset
DELETE /v1/rwkv/sessions/{session_id}
POST   /v1/rwkv/sessions/evict
```

`VLLM_RWKV_SESSION_TTL_SECONDS` controls lazy reclamation of detached sessions
and defaults to 1800 seconds. The explicit `evict` endpoint can be used by a
supervisor or scheduler. Active sessions are never reclaimed.

Current limitations:

- Only RWKV7 text requests are supported; multimodal and LoRA stateful chunks
  are rejected.
- Sampling parameters must remain unchanged within a session.
- Stateful requests require `n=1` and do not support stop strings. The server
  removes model-default stop strings for stateful generation because the
  streaming-input engine cannot apply them; use `max_completion_tokens` or
  `stop_token_ids` for an explicit bound.
- State is process-local and GPU-resident. It is not persisted across process
  restart and cannot be shared across workers. A continuation after restart
  returns HTTP 410; the client must replay the full conversation with
  `rwkv_session_action="create"`.
- Session rows are bounded by the configured state pool (`max_num_seqs`); a
  deployment must size this for its maximum number of live sessions.
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
