"""Measure RWKV stateful streaming against the normal sliding-window path."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM


def _token_ids(length: int) -> list[int]:
    """Build a deterministic token sequence without a tokenizer round-trip."""
    return [100 + (idx % 97) for idx in range(length)]


async def _collect(generator):
    first_token_at: float | None = None
    output = None
    started_at = time.perf_counter()
    async for item in generator:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        output = item
    finished_at = time.perf_counter()
    return output, {
        "time_to_first_output_s": (
            None if first_token_at is None else first_token_at - started_at
        ),
        "total_time_s": finished_at - started_at,
    }


async def _run_one(
    engine: AsyncLLM,
    token_ids: list[int],
    mode: str,
    chunk_size: int,
    request_id: str,
) -> dict:
    params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1,
    )
    async def inputs():
        for start in range(0, len(token_ids), chunk_size):
            yield StreamingInput(
                prompt=TokensPrompt(
                    prompt_token_ids=token_ids[start : start + chunk_size]
                ),
                session_id=request_id if mode == "stateful" else None,
                context_mode=mode,
            )

    prompt = inputs()

    output, timing = await _collect(
        engine.generate(prompt, params, request_id=request_id)
    )
    if output is None:
        raise RuntimeError(f"No output received for {request_id}")
    generated = output.outputs[0]
    return {
        "request_id": request_id,
        "context_mode": mode,
        "total_history_tokens": len(token_ids),
        "chunk_size": chunk_size,
        "output_token_ids": generated.token_ids,
        "output_text": generated.text,
        **timing,
    }


async def _main(args: argparse.Namespace) -> None:
    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        context_window=args.context_window,
        context_window_strategy="sliding_window",
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    results = []
    try:
        for length in args.lengths:
            token_ids = _token_ids(length)
            results.append(
                await _run_one(
                    engine,
                    token_ids,
                    "sliding",
                    args.chunk_size,
                    f"sliding-{length}",
                )
            )
            results.append(
                await _run_one(
                    engine,
                    token_ids,
                    "stateful",
                    args.chunk_size,
                    f"stateful-{length}",
                )
            )
    finally:
        engine.shutdown()

    payload = {
        "model": args.model,
        "context_window": args.context_window,
        "lengths": args.lengths,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 12288, 20480])
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
