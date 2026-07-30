#!/usr/bin/env python3
"""Probe generated-output quality for the RWKV stateful API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def request(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
) -> tuple[int | None, dict]:
    body = None if payload is None else json.dumps(payload).encode()
    request_obj = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=90) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {"http_error": error.read().decode("utf-8", "replace")}
    except Exception as error:
        return None, {"exception": repr(error)}


def message_content(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def finish_reason(response: dict) -> str | None:
    try:
        return response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def chat_payload(model: str, content: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def run_probe(base_url: str, model: str, api_key: str) -> tuple[bool, list[dict]]:
    results: list[dict] = []

    status, response = request(
        "POST",
        f"{base_url}/v1/chat/completions",
        api_key,
        {
            **chat_payload(model, "Answer exactly OK and nothing else.", 64),
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the requested output format exactly.",
                },
                {"role": "user", "content": "Answer exactly OK and nothing else."},
            ],
        },
    )
    output = message_content(response)
    results.append(
        {
            "check": "instruction",
            "status": status,
            "pass": output.strip() == "OK",
            "finish_reason": finish_reason(response),
            "output": output[:500],
        }
    )

    status, response = request(
        "POST",
        f"{base_url}/v1/chat/completions",
        api_key,
        chat_payload(model, "Give the final answer clearly: 17 + 25 = ?", 128),
    )
    output = message_content(response)
    results.append(
        {
            "check": "arithmetic",
            "status": status,
            "pass": "42" in output,
            "finish_reason": finish_reason(response),
            "output": output[:500],
        }
    )

    session_id = f"quality-{uuid.uuid4().hex[:12]}"
    status_create, create = request(
        "POST",
        f"{base_url}/v1/chat/completions",
        api_key,
        {
            **chat_payload(
                model, "Remember this exact token: MARKER_7391. Reply ACK.", 64
            ),
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )
    status_continue, continuation = request(
        "POST",
        f"{base_url}/v1/chat/completions",
        api_key,
        {
            **chat_payload(
                model,
                "What exact token did I ask you to remember? Output only the token.",
                64,
            ),
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "continue",
        },
    )
    status_delete, _ = request(
        "DELETE",
        f"{base_url}/v1/rwkv/sessions/{session_id}",
        api_key,
    )
    output = message_content(continuation)
    results.append(
        {
            "check": "stateful_memory",
            "create_status": status_create,
            "continuation_status": status_continue,
            "delete_status": status_delete,
            "pass": status_continue == 200 and "MARKER_7391" in output,
            "create_output": message_content(create)[:240],
            "continuation_output": output[:500],
            "error": continuation.get("http_error") or continuation.get("exception"),
        }
    )

    status, response = request(
        "POST",
        f"{base_url}/v1/chat/completions",
        api_key,
        {
            **chat_payload(model, "Call get_weather for Shanghai.", 96),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )
    try:
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        passed = (
            status == 200
            and bool(tool_calls)
            and tool_calls[0]["function"]["name"] == "get_weather"
            and arguments.get("city") == "Shanghai"
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        tool_calls = []
        passed = False
    results.append(
        {
            "check": "tool_call",
            "status": status,
            "pass": passed,
            "tool_calls": tool_calls,
        }
    )
    return all(result["pass"] for result in results), results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="rwkv-stateful")
    args = parser.parse_args()

    passed, results = run_probe(args.base_url, args.model, args.api_key)
    print(
        json.dumps(
            {
                "model": args.model,
                "summary": "PASS" if passed else "QUALITY_ISSUE",
                "results": results,
            },
            ensure_ascii=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
