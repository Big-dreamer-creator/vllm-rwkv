from __future__ import annotations

"""Compare the four knowledge-runner prompt profiles across long contexts."""

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vllm.tokenizers.rwkv import RWKVTokenizer


TARGET_TOKENS = (1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480)
FILLER_UNIT = "0123456789 "
ANSWERS = {
    "early": "EARLY-RWKV-314159",
    "late": "LATE-RWKV-271828",
}

PROMPT_TEMPLATES = {
    "normal_direct": (
        "User: You are a very talented expert in <SUBJECT>. Answer this question:\n"
        "<Q>\n"
        "<CHOICES>\n\n"
        "Assistant: The answer is"
    ),
    "normal_cot": (
        "User✿You are a very talented expert in <SUBJECT>. Answer this question:\n"
        "<Q>\n"
        "<CHOICES>✿\n"
        "Bot✿<think"
    ),
    "naive_direct": "User: <Q>\n<CHOICES>\n\nAssistant:",
    "naive_cot": "User: <Q>\n<CHOICES>\n\nAssistant: <think",
}


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


def _question(placement: str, answer: str, filler_repeats: int) -> str:
    filler = FILLER_UNIT * filler_repeats
    marker = f"The exact answer token is {answer}."
    if placement == "early":
        context = f"{marker}\nUnrelated archive text:\n{filler}\n"
    else:
        context = f"Unrelated archive text:\n{filler}\n{marker}\n"
    return (
        "task: recover the exact answer token from the context below.\n"
        f"{context}"
        "Question: What is the exact answer token? Return only that token."
    )


def _choices() -> str:
    return (
        "A. The context contains a token.\n"
        "B. The context contains a hidden answer.\n"
        "C. The context contains an archive marker.\n"
        "D. The context contains a task instruction."
    )


def _build_prompt(
    tokenizer: RWKVTokenizer,
    target: int,
    placement: str,
    format_name: str,
) -> tuple[str, int, int]:
    answer = ANSWERS[placement]
    template = PROMPT_TEMPLATES[format_name]

    def make(repeats: int) -> str:
        prompt = template.replace("<SUBJECT>", "context-window retention")
        prompt = prompt.replace("<Q>", _question(placement, answer, repeats))
        return prompt.replace("<CHOICES>", _choices())

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
    for format_name in PROMPT_TEMPLATES:
        for target in TARGET_TOKENS:
            for placement in ("early", "late"):
                prompt, raw_tokens, filler_repeats = _build_prompt(
                    tokenizer, target, placement, format_name
                )
                expected = ANSWERS[placement]
                row: dict[str, Any] = {
                    "format": format_name,
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
                    output = str(choice.get("text") or "")
                    row.update(
                        {
                            "actual_prompt_tokens": response.get("usage", {}).get(
                                "prompt_tokens"
                            ),
                            "total_tokens": response.get("usage", {}).get(
                                "total_tokens"
                            ),
                            "finish_reason": choice.get("finish_reason"),
                            "output": output,
                            "correct": bool(re.search(re.escape(expected), output)),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve every point
                    row.update({"error": repr(exc), "correct": False})
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "formats": list(PROMPT_TEMPLATES),
        "targets": TARGET_TOKENS,
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
