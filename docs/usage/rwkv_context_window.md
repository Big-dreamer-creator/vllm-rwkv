# RWKV inference-side context windows

RWKV serving keeps the model's `max_model_len` separate from the prompt
window used for a request. This allows current checkpoints to run with one
8192-token window while leaving larger-context checkpoints available later.

```bash
vllm serve /path/to/rwkv7-checkpoint.pth \
  --tokenizer-mode rwkv \
  --max-model-len 10240 \
  --context-window 8192 \
  --context-window-strategy sliding_window
```

`sliding_window` keeps the newest prompt tokens. For RWKV, older complete chat
turns are dropped before rendering when possible, then token-level left
truncation is used as the final safety boundary. The requested output budget is
reserved, so the effective total remains within `max_model_len`.

To disable inference-side truncation while retaining the model's own limit:

```bash
vllm serve /path/to/rwkv7-checkpoint.pth \
  --tokenizer-mode rwkv \
  --max-model-len 10240 \
  --context-window 8192 \
  --context-window-strategy none
```

If `--context-window` is omitted for an RWKV7 checkpoint, the policy uses the
checkpoint's advertised `context_window` or `rwkv_context_window`; otherwise
the current default is 8192. An explicit `--context-window` always wins.
