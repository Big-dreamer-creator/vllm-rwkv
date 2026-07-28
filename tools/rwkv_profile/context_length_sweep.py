from __future__ import annotations

"""Sweep RWKV raw completion prompts from 1K to 20K input tokens."""

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vllm.tokenizers.rwkv import RWKVTokenizer


ANSWER_TOKENS = {
    "early": "EARLY-RWKV-314159",
    "late": "LATE-RWKV-271828",
}
TARGET_TOKENS = (1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480)
FILLER_UNIT = "0123456789 "


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _build_prompt(tokenizer: RWKVTokenizer, target: int, placement: str) -> tuple[str, int, int]:
    answer = ANSWER_TOKENS[placement]

    def make(repeats: int) -> str:
        filler = FILLER_UNIT * repeats
        if placement == "early":
            task = (
                f"The exact answer token is {answer}.\n"
                f"Unrelated archive text:\n{filler}\n"
            )
        else:
            task = (
                f"Unrelated archive text:\n{filler}\n"
                f"The exact answer token is {answer}.\n"
            )
        return (
            "User: task\n\n"
            "Read the task context and answer the question at the end. "
            "Return only the exact answer token.\n"
            f"{task}"
            "Question: What is the exact answer token?\n\n"
            "Assistant: <think"
        )

    low, high = 0, max(1, target * 2)
    while len(tokenizer.encode(make(high))) < target:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if len(tokenizer.encode(make(middle))) < target:
            low = middle
        else:
            high = middle
    candidates = [max(0, low - 1), low, high]
    prompt = min(
        (make(repeats) for repeats in candidates),
        key=lambda value: abs(len(tokenizer.encode(value)) - target),
    )
    return prompt, len(tokenizer.encode(prompt)), prompt.count(FILLER_UNIT)


def _completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    timeout_s: float,
    seed: int,
) -> dict[str, Any]:
    return _post_json(
        f"{base_url.rstrip('/')}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "stream": False,
        },
        api_key=api_key,
        timeout_s=timeout_s,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = RWKVTokenizer.from_pretrained(args.model)
    results: list[dict[str, Any]] = []
    for target in TARGET_TOKENS:
        for placement in ("early", "late"):
            prompt, raw_tokens, filler_repeats = _build_prompt(
                tokenizer, target, placement
            )
            expected = ANSWER_TOKENS[placement]
            row: dict[str, Any] = {
                "target_tokens": target,
                "placement": placement,
                "raw_prompt_tokens": raw_tokens,
                "filler_repeats": filler_repeats,
                "expected": expected,
            }
            try:
                response = _completion(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    api_key=args.api_key,
                    timeout_s=args.timeout_s,
                    seed=args.seed,
                )
                choice = response["choices"][0]
                text = str(choice.get("text") or "")
                row.update(
                    {
                        "actual_prompt_tokens": response.get("usage", {}).get(
                            "prompt_tokens"
                        ),
                        "total_tokens": response.get("usage", {}).get("total_tokens"),
                        "finish_reason": choice.get("finish_reason"),
                        "output": text,
                        "correct": bool(re.search(re.escape(expected), text)),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep every failed point
                row.update({"error": repr(exc), "correct": False})
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "format": "User: task\\n\\n...\\nAssistant: <think",
        "max_tokens": 128,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="new-vllm")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
