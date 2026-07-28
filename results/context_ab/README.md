# RWKV context-window A/B results

These JSON files are generated from the same weights, request payloads, seed,
and sampling settings. `none32` is the no-truncation baseline and `sliding32`
uses an 8192-token inference-side sliding window.

The benchmark runner is
`tools/rwkv_profile/compare_context_window.py`. It uses 32 real MMLU-SR rows,
32 real GSM8K rows, and three deterministic long-context retrieval cases per
model.
