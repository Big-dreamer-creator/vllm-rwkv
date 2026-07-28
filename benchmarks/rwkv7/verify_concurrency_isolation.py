#!/usr/bin/env python3
"""Verify request isolation and sampling stability on an OpenAI service."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--greedy-repeat", type=int, default=4)
    parser.add_argument("--seeded-repeat", type=int, default=1)
    parser.add_argument("--rapid-concurrency", type=int, default=128)
    return parser.parse_args()


def make_prompts() -> list[str]:
    repeat_counts = (8, 64, 256, 1024, 4096, 8000)
    return [
        (
            f"Request identity {index}. Continue this text consistently.\n"
            + " benchmark" * repeat_count
        )
        for index, repeat_count in enumerate(repeat_counts)
    ]


def main() -> int:
    args = parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    endpoint = f"{args.base_url.rstrip('/')}/completions"
    timeout = httpx.Timeout(args.timeout_s)

    def run_checks() -> dict[str, object]:

        def complete(
            prompt: str,
            *,
            max_tokens: int,
            temperature: float,
            top_p: float,
            seed: int | None = None,
        ) -> Completion:
            payload: dict[str, object] = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "ignore_eos": True,
            }
            if seed is not None:
                payload["seed"] = seed
            with httpx.Client(headers=headers, timeout=timeout) as request_client:
                response = request_client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            return Completion(
                text=str(body["choices"][0]["text"]),
                finish_reason=body["choices"][0].get("finish_reason"),
                completion_tokens=int(body["usage"]["completion_tokens"]),
            )

        prompts = make_prompts()
        greedy_baseline = {
            prompt: complete(
                prompt,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
            )
            for prompt in prompts
        }

        for wave in range(args.waves):
            batch = prompts * args.greedy_repeat
            random.Random(1729 + wave).shuffle(batch)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(batch)
            ) as executor:
                outputs = list(
                    executor.map(
                        lambda prompt: complete(
                            prompt,
                            max_tokens=64,
                            temperature=0.0,
                            top_p=1.0,
                        ),
                        batch,
                    )
                )
            mismatches = [
                index
                for index, (prompt, output) in enumerate(
                    zip(batch, outputs, strict=True)
                )
                if output != greedy_baseline[prompt]
            ]
            if mismatches:
                raise RuntimeError(
                    f"greedy isolation mismatch in wave {wave}: {mismatches[:8]}"
                )

        seeds = [2718 + index for index in range(len(prompts))]
        seeded_baseline = [
            complete(
                prompt,
                max_tokens=64,
                temperature=0.8,
                top_p=0.3,
                seed=seed,
            )
            for prompt, seed in zip(prompts, seeds, strict=True)
        ]
        seeded_items = list(zip(prompts, seeds, strict=True)) * args.seeded_repeat
        random.Random(3141).shuffle(seeded_items)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(seeded_items)
        ) as executor:
            seeded_concurrent = list(
                executor.map(
                    lambda item: complete(
                        item[0],
                        max_tokens=64,
                        temperature=0.8,
                        top_p=0.3,
                        seed=item[1],
                    ),
                    seeded_items,
                )
            )
        seeded_mismatches = [
            index
            for index, (item, actual) in enumerate(
                zip(seeded_items, seeded_concurrent, strict=True)
            )
            if actual != seeded_baseline[prompts.index(item[0])]
        ]
        if seeded_mismatches:
            raise RuntimeError(f"seeded isolation mismatch: {seeded_mismatches[:8]}")

        rapid_prompts = [
            prompts[index % (len(prompts) - 1)]
            for index in range(args.rapid_concurrency)
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.rapid_concurrency
        ) as executor:
            rapid_outputs = list(
                executor.map(
                    lambda prompt: complete(
                        prompt,
                        max_tokens=64,
                        temperature=0.8,
                        top_p=0.3,
                    ),
                    rapid_prompts,
                )
            )
        invalid_rapid = [
            index
            for index, output in enumerate(rapid_outputs)
            if output.completion_tokens != 64 or output.finish_reason != "length"
        ]
        if invalid_rapid:
            raise RuntimeError(
                f"rapid sampling returned invalid completions: {invalid_rapid[:8]}"
            )

        return {
            "status": "ok",
            "model": args.model,
            "greedy_prompts": len(prompts),
            "greedy_waves": args.waves,
            "greedy_requests_per_wave": len(prompts) * args.greedy_repeat,
            "seeded_requests": len(seeded_items),
            "rapid_requests": args.rapid_concurrency,
        }

    result = run_checks()
    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
