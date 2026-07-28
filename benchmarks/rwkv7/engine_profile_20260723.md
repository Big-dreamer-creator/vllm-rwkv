# RWKV7 inference engine profile — 2026-07-23

## Scope

After profiling on four evaluation instances, all eight vLLM instances were
released and redeployed with the selected configuration:

| Host | GPU | Model | Port |
| --- | ---: | --- | ---: |
| 157 | 0 | 1.5B | 19315 |
| 157 | 1 | 1.5B | 19316 |
| 157 | 2 | 2.9B | 19329 |
| 157 | 3 | 2.9B | 19330 |
| 8222 | 0 | 7.2B | 18072 |
| 8222 | 1 | 13.3B | 18133 |
| 8222 | 2 | 13.3B | 18134 |
| 8222 | 3 | 7.2B | 18073 |

Each model therefore has two independent single-GPU instances. The deployment
does not use tensor parallelism between the pair.

## Selected configuration

```text
VLLM_USE_V2_MODEL_RUNNER=1
--gpu-memory-utilization 0.97
--max-model-len 10240
--max-num-batched-tokens 49152
--max-num-seqs 1024
```

The existing RWKV tokenizer mode, model paths, ports, API key, and default
CUDA graph capture sizes were retained.

`max-num-batched-tokens=49152` is the selected safe performance point. For
eight 8K–9.7K prompts it reduces prefill scheduling from three chunks to two.
Increasing it to 65536 does not reduce the chunk count further because the
generated prompt contains one additional token. The 65536 setting also raised
idle memory to approximately 92 GB on 13.3B, leaving only about 6 GB of
headroom, so it was rejected.

CUDA graph capture was not extended from 512 to 1024. The observed evaluation
clients normally use 48–128 workers; increasing graph coverage would consume
more memory without evidence of a useful gain.

## Long-context results

All measurements below completed with zero request failures.

At eight concurrent prefill requests:

| Model | Context | 32768 input tok/s | 49152 input tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 9728 | 35061.9 | 36017.5 | +2.7% |
| 2.9B | 9728 | 19273.5 | 19700.7 | +2.2% |
| 7.2B | 9728 | 10544.0 | 10696.5 | +1.4% |
| 13.3B | 9728 | 5656.1 | 5836.6 | +3.2% |

The 8192-token measurements were within normal run-to-run variance for 1.5B
and improved by roughly 0.4–1.0% for the other models.

For 1024-token input and 4096-token output at concurrency 64, output
throughput was:

| Model | Output tok/s |
| --- | ---: |
| 1.5B | 10753.2 |
| 2.9B | 5875.7 |
| 7.2B | 3562.3 |
| 13.3B | 1944.4 |

These results are effectively unchanged from the 32768 configuration.

The exact 10240-token boundary was tested as 6143 input tokens plus one
generated prompt token plus 4096 output tokens. At concurrency 16, output
throughput was 3241.7, 1774.8, 1094.7, and 643.1 tok/s for 1.5B, 2.9B, 7.2B,
and 13.3B respectively, with zero failures.

## Recommended client concurrency

Engine capacity remains `max-num-seqs=1024`; these are recommended evaluation
worker counts, not engine limits.

| Workload | 1.5B | 2.9B | 7.2B | 13.3B |
| --- | ---: | ---: | ---: | ---: |
| Normal evaluation | 128 | 128 | 128 | 128 |
| Throughput evaluation | 128 | up to 512 | 256 | 256 |
| Long generation (up to 4096 output) | 64–128 | 64–128 | 64 | 64 |

For a 1024-token input and 512-token output sweep, concurrency 512 still had
zero failures on every model. It is not recommended for 7.2B or 13.3B:
compared with concurrency 256, throughput increased only 1.7% and 4.6% while
median end-to-end latency rose to approximately 76 and 140 seconds.

## Correctness and concurrency gates

The service-level isolation probe performs:

- serial greedy baselines followed by shuffled heterogeneous concurrent
  requests;
- exact comparison of text, finish reason, and completion-token count;
- serial versus concurrent comparison with explicit per-request seeds;
- a rapid unseeded sampling path that checks response length and finish reason.

Each model passed:

- 528 exact greedy concurrent comparisons;
- 96 exact seeded concurrent comparisons;
- 256 rapid unseeded concurrent requests.

After the all-eight redeployment, every instance passed an additional
service-level probe containing six heterogeneous prompts up to approximately
8K tokens, six exact seeded comparisons, and 16 rapid unseeded requests. One
instance of each model also passed the exact 10240-token boundary again with
6143 input tokens and 4096 output tokens.

The targeted RWKV lifecycle, CUDA graph slot, state-row reuse, prefill/decode
transition, and rapid-sampler regression selection passed 10/10.

The broader local test run was not a clean environment-wide pass: 34
FlashInfer cases require a usable sm75+ GPU in the local test environment, and
one pre-existing expectation still assumes the old four-row model-state shape
instead of the intentional padded fifth row. Neither issue was introduced or
modified during this profile.

## Runtime state

After redeployment and validation, approximate GPU allocations were:

| Host/GPU | Model | Used memory | Device memory |
| --- | --- | ---: | ---: |
| 157/GPU0 | 1.5B | 12.0 GB | 49.1 GB |
| 157/GPU1 | 1.5B | 12.0 GB | 49.1 GB |
| 157/GPU2 | 2.9B | 19.5 GB | 49.1 GB |
| 157/GPU3 | 2.9B | 19.5 GB | 49.1 GB |
| 8222/GPU0 | 7.2B | 35.1 GB | 97.9 GB |
| 8222/GPU1 | 13.3B | 62.5 GB | 97.9 GB |
| 8222/GPU2 | 13.3B | 62.5 GB | 97.9 GB |
| 8222/GPU3 | 7.2B | 35.1 GB | 97.9 GB |

Final health checks passed on all eight direct ports. The selected
configuration logs contained no traceback, CUDA error, OOM, or engine
initialization error.

## 8222 forwarding

The 8222 services are forwarded to loopback ports on 157:

| 157 forwarded port | 8222 target | Model |
| ---: | --- | --- |
| 29572 | 127.0.0.1:18072 | 7.2B |
| 29573 | 127.0.0.1:18073 | 7.2B |
| 29533 | 127.0.0.1:18133 | 13.3B |
| 29534 | 127.0.0.1:18134 | 13.3B |

The forwarding process runs under the detached screen
`g1h_8222_forwards_20260721_watchdog`. It uses
`ExitOnForwardFailure=yes`, a 15-second server-alive interval, three missed
keepalives, and a five-second restart loop. Its self-recovery was tested by
terminating the SSH child: a new child and all four model routes recovered
within the configured restart window.

## Rollback

If a workload exposes a problem not covered by this matrix, restart the
affected single-GPU instance with:

```text
--max-num-batched-tokens 32768
```

Keep all other selected parameters unchanged. This is the measured
current-code baseline and does not require an engine source rollback.
