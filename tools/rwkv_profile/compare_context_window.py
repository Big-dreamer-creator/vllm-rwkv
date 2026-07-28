from __future__ import annotations

"""Run a deterministic accuracy A/B for RWKV context-window handling.

The script deliberately uses the same request payloads and sampling settings for
both services.  It combines real, locally available MMLU-SR/GSM8K rows with
small long-context retrieval cases whose expected answer is unambiguous.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _http_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    api_key: str,
    timeout_s: float,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }
    return _http_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload,
        api_key=api_key,
        timeout_s=timeout_s,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _even_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    return [rows[(index * len(rows)) // count] for index in range(count)]


def _mcq_cases(path: Path, count: int) -> list[dict[str, Any]]:
    cases = []
    for index, row in enumerate(_even_sample(_load_jsonl(path), count)):
        choices = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
        prompt = (
            "Answer the multiple-choice question. Return only one uppercase "
            "letter (A, B, C, or D), with no explanation.\n\n"
            f"Question: {row['question']}\nChoices:\n{choices}"
        )
        cases.append(
            {
                "id": f"mmlu_sr_{index}",
                "group": "real_mmlu_sr",
                "kind": "mcq",
                "messages": [{"role": "user", "content": prompt}],
                "expected": str(row["answer"]).strip().upper(),
                "max_tokens": 64,
            }
        )
    return cases


def _gsm_cases(path: Path, count: int) -> list[dict[str, Any]]:
    cases = []
    for index, row in enumerate(_even_sample(_load_jsonl(path), count)):
        prompt = (
            "Solve the following math word problem. Return only the final "
            "numeric answer, with no explanation.\n\n"
            f"Problem: {row['question']}"
        )
        cases.append(
            {
                "id": f"gsm8k_{index}",
                "group": "real_gsm8k",
                "kind": "number",
                "messages": [{"role": "user", "content": prompt}],
                "expected": str(row["answer"]).strip(),
                "max_tokens": 64,
            }
        )
    return cases


def _long_cases() -> list[dict[str, Any]]:
    filler = "0123456789 " * 1800
    within = "0123456789 " * 1200
    return [
        {
            "id": "long_recent_needle",
            "group": "window_retrieval",
            "kind": "needle",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Irrelevant archive text:\n{filler}\n"
                        "The exact secret code is RECENT-BLUE-731.\n"
                        "What is the exact secret code? Return only the code."
                    ),
                }
            ],
            "expected": "RECENT-BLUE-731",
            "max_tokens": 32,
        },
        {
            "id": "long_early_needle",
            "group": "window_loss_tradeoff",
            "kind": "needle",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "The exact secret code is EARLY-RED-482.\n"
                        f"Irrelevant archive text:\n{filler}\n"
                        "What is the exact secret code? Return only the code."
                    ),
                }
            ],
            "expected": "EARLY-RED-482",
            "max_tokens": 32,
        },
        {
            "id": "long_within_window_needle",
            "group": "window_retrieval",
            "kind": "needle",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Irrelevant archive text:\n{within}\n"
                        "The exact secret code is WITHIN-GREEN-916.\n"
                        "What is the exact secret code? Return only the code."
                    ),
                }
            ],
            "expected": "WITHIN-GREEN-916",
            "max_tokens": 32,
        },
    ]


def _normalize_number(value: str) -> str:
    return value.replace(",", "").strip().lstrip("+")


def _score(case: dict[str, Any], content: str) -> tuple[bool, str]:
    kind = case["kind"]
    if kind == "mcq":
        matches = re.findall(r"(?i)(?<![A-Z])([A-D])(?![A-Z])", content)
        prediction = matches[-1].upper() if matches else ""
        return prediction == case["expected"], prediction
    if kind == "number":
        matches = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", content)
        prediction = _normalize_number(matches[-1]) if matches else ""
        return prediction == _normalize_number(case["expected"]), prediction
    prediction = case["expected"] if case["expected"] in content else ""
    return bool(prediction), prediction


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = _mcq_cases(Path(args.mmlu_path), args.dataset_count)
    cases.extend(_gsm_cases(Path(args.gsm8k_path), args.dataset_count))
    cases.extend(_long_cases())
    results = []
    for index, case in enumerate(cases):
        started = time.monotonic()
        row: dict[str, Any] = {
            "id": case["id"],
            "group": case["group"],
            "kind": case["kind"],
            "expected": case["expected"],
        }
        try:
            response = _chat(
                base_url=args.base_url,
                model=args.model,
                messages=case["messages"],
                max_tokens=case["max_tokens"],
                api_key=args.api_key,
                timeout_s=args.timeout_s,
                seed=args.seed,
            )
            choice = response["choices"][0]
            message = choice.get("message", {})
            content = str(message.get("content") or "")
            correct, prediction = _score(case, content)
            row.update(
                {
                    "correct": correct,
                    "prediction": prediction,
                    "content": content,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage", {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain per-case evidence
            row.update({"error": repr(exc), "correct": False})
        row["latency_s"] = round(time.monotonic() - started, 3)
        results.append(row)
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "total": len(cases),
                    "id": case["id"],
                    "correct": row.get("correct"),
                    "prompt_tokens": row.get("usage", {}).get("prompt_tokens"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "case_count": len(cases),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--api-key", default="new-vllm")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dataset-count", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--mmlu-path",
        default="/home/rwkv/chase/rwkv-skills/data/mmlu_sr_question_and_answer/test.jsonl",
    )
    parser.add_argument(
        "--gsm8k-path",
        default="/home/rwkv/chase/rwkv-skills/data/gsm8k/test.jsonl",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": report["case_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
