from __future__ import annotations

"""Repeat short prompts and probe very long prompts under a context policy."""

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vllm.tokenizers.rwkv import RWKVTokenizer


FILLER_UNIT = "0123456789 "
FORMATS = ("normal_cot", "naive_cot")
SHORT_TARGETS = (4096, 8192)
LONG_TARGETS = (12288, 16384, 20480, 32768)
PROMPT_TEMPLATES = {
    "normal_cot": (
        "User✿You are a very talented expert in <SUBJECT>. Answer this question:\n"
        "<Q>\n"
        "<CHOICES>✿\n"
        "Bot✿<think"
    ),
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


def _render_prompt(
    tokenizer: RWKVTokenizer,
    target: int,
    placement: str,
    format_name: str,
) -> tuple[str, int]:
    template = PROMPT_TEMPLATES[format_name]
    expected = "C"

    def make(repeats: int) -> str:
        filler = FILLER_UNIT * repeats
        signal = f"The context signal says the correct choice is {expected}."
        if placement == "early":
            context = f"{signal}\nUnrelated archive text:\n{filler}\n"
        else:
            context = f"Unrelated archive text:\n{filler}\n{signal}\n"
        question = (
            "task\n\n"
            "Read the context and answer the final question.\n"
            f"{context}"
            "Question: Which choice is indicated by the context? "
            "Return only the choice letter."
        )
        choices = (
            "A. The first option.\n"
            "B. The second option.\n"
            "C. The third option.\n"
            "D. The fourth option."
        )
        prompt = template.replace("<SUBJECT>", "context-window retention")
        prompt = prompt.replace("<Q>", question).replace("<CHOICES>", choices)
        return prompt

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
    return prompt, len(tokenizer.encode(prompt))


def _generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    timeout_s: float,
    seed: int,
) -> dict[str, Any]:
    response = _post_json(
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
    choice = response["choices"][0]
    output = str(choice.get("text") or "")
    return {
        "actual_prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
        "total_tokens": response.get("usage", {}).get("total_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "output": output,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest()[:16],
        "correct": bool(
            re.search(r"(?:answer|choice|option)\s*(?:is|:)\s*C\b", output, re.I)
            or re.search(r"(?:^|\n)\s*C(?:[.)]|\s*$)", output, re.I)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = RWKVTokenizer.from_pretrained(args.tokenizer)
    if args.target_set == "short":
        targets = SHORT_TARGETS
    elif args.target_set == "long":
        targets = LONG_TARGETS
    else:
        targets = SHORT_TARGETS + LONG_TARGETS
    repeats = args.repeats if args.repeats > 0 else 1
    results: list[dict[str, Any]] = []
    for format_name in FORMATS:
        for target in targets:
            for placement in ("early", "late"):
                prompt, raw_tokens = _render_prompt(
                    tokenizer, target, placement, format_name
                )
                for repeat_index in range(repeats):
                    row: dict[str, Any] = {
                        "format": format_name,
                        "target_tokens": target,
                        "placement": placement,
                        "repeat_index": repeat_index,
                        "raw_prompt_tokens": raw_tokens,
                    }
                    try:
                        row.update(
                            _generate(
                                base_url=args.base_url,
                                model=args.model,
                                prompt=prompt,
                                api_key=args.api_key,
                                timeout_s=args.timeout_s,
                                seed=args.seed,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve each probe
                        row.update({"error": repr(exc), "correct": False})
                    results.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeats": repeats,
        "formats": FORMATS,
        "targets": targets,
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
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--api-key", default="new-vllm")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-set", choices=("short", "long", "all"), default="all")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
