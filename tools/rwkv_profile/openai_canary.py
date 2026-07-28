"""Exercise the RWKV OpenAI API and session lifecycle over HTTP."""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    status_code: int
    elapsed_s: float
    body: dict


def _headers(api_key: str | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def _post_chat(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> RequestResult:
    started = time.perf_counter()
    response = await client.post(url, json=payload)
    elapsed_s = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {"text": response.text}
    return RequestResult(response.status_code, elapsed_s, body)


def _chat_payload(model: str, index: int) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Return the integer {index} and nothing else.",
            }
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "seed": 42,
    }


async def _load_test(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    requests: int,
    concurrency: int,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int) -> RequestResult:
        async with semaphore:
            return await _post_chat(client, url, _chat_payload(model, index))

    return await asyncio.gather(*(run(index) for index in range(requests)))


def _load_summary(results: list[RequestResult]) -> dict:
    elapsed = sorted(result.elapsed_s for result in results)
    successes = [result for result in results if result.status_code == 200]
    p95_index = min(len(elapsed) - 1, max(0, int(len(elapsed) * 0.95) - 1))
    return {
        "requests": len(results),
        "successes": len(successes),
        "errors": len(results) - len(successes),
        "min_s": min(elapsed),
        "mean_s": statistics.mean(elapsed),
        "p95_s": elapsed[p95_index],
        "max_s": max(elapsed),
    }


async def _session_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_id: str,
) -> dict:
    chat_url = f"{base_url}/v1/chat/completions"
    session_url = f"{base_url}/v1/rwkv/sessions/{session_id}"
    create = await _post_chat(
        client,
        chat_url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "Say alpha."},
            ],
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )
    continuation = await _post_chat(
        client,
        chat_url,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Now say beta."}],
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "continue",
        },
    )
    if create.status_code != 200 or continuation.status_code != 200:
        return {
            "create_status": create.status_code,
            "continuation_status": continuation.status_code,
            "create_response": create.body,
            "continuation_response": continuation.body,
        }
    status_before_reset = await client.get(session_url)
    reset = await client.post(f"{session_url}/reset")
    status_after_reset = await client.get(session_url)
    delete = await client.delete(session_url)
    status_after_delete = await client.get(session_url)
    return {
        "create_status": create.status_code,
        "continuation_status": continuation.status_code,
        "status_before_reset": status_before_reset.status_code,
        "reset_status": reset.status_code,
        "processed_after_reset": status_after_reset.json().get("processed_token_count"),
        "delete_status": delete.status_code,
        "status_after_delete": status_after_delete.status_code,
        "create_response": create.body,
        "continuation_response": continuation.body,
    }


async def _tool_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
) -> dict:
    result = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Call get_weather for Shanghai. Do not answer directly.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0.0,
            "seed": 42,
        },
    )
    choice = result.body.get("choices", [{}])[0]
    message = choice.get("message", {})
    return {
        "status": result.status_code,
        "elapsed_s": result.elapsed_s,
        "tool_calls": message.get("tool_calls"),
        "content": message.get("content"),
        "response": result.body,
    }


async def _ttl_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_id: str,
    wait_s: float,
) -> dict:
    create = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say ready."}],
            "max_tokens": 1,
            "temperature": 0.0,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )
    await asyncio.sleep(wait_s)
    evict = await client.post(f"{base_url}/v1/rwkv/sessions/evict")
    status = await client.get(f"{base_url}/v1/rwkv/sessions/{session_id}")
    return {
        "create_status": create.status_code,
        "evict_status": evict.status_code,
        "evict_response": evict.json(),
        "status_after_evict": status.status_code,
    }


async def _stateful_tool_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_id: str,
) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    create = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Call get_weather for Shanghai."}],
            "tools": [tool],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )
    tool_calls = (
        create.body.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    )
    continuation = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"] if tool_calls else "missing",
                    "content": '{"temperature": 28, "condition": "sunny"}',
                },
            ],
            "tools": [tool],
            "tool_choice": "none",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "continue",
        },
    )
    delete = await client.delete(f"{base_url}/v1/rwkv/sessions/{session_id}")
    return {
        "create_status": create.status_code,
        "create_tool_calls": tool_calls,
        "continuation_status": continuation.status_code,
        "continuation_response": continuation.body,
        "delete_status": delete.status_code,
    }


async def _recovery_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_id: str,
) -> dict:
    continue_request = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Now say gamma."}],
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "continue",
        },
    )
    replay = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say ready."}],
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": 42,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )
    status = await client.get(f"{base_url}/v1/rwkv/sessions/{session_id}")
    delete = await client.delete(f"{base_url}/v1/rwkv/sessions/{session_id}")
    return {
        "continue_after_restart_status": continue_request.status_code,
        "continue_after_restart_response": continue_request.body,
        "replay_create_status": replay.status_code,
        "replay_create_response": replay.body,
        "status_after_replay": status.status_code,
        "delete_status": delete.status_code,
    }


async def _stateful_soak(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_count: int,
    turns: int,
    concurrency: int,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    session_ids = [f"soak-{int(time.time())}-{index}" for index in range(session_count)]

    async def run_session(session_id: str) -> None:
        for turn in range(turns):
            async with semaphore:
                result = await _post_chat(
                    client,
                    f"{base_url}/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Session turn {turn}: reply with ready.",
                            }
                        ],
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "seed": 42,
                        "rwkv_context_mode": "stateful",
                        "rwkv_session_id": session_id,
                        "rwkv_session_action": "create" if turn == 0 else "continue",
                    },
                )
                results.append(result)
            if result.status_code != 200:
                break

    await asyncio.gather(*(run_session(session_id) for session_id in session_ids))
    delete_statuses = []
    for session_id in session_ids:
        response = await client.delete(f"{base_url}/v1/rwkv/sessions/{session_id}")
        delete_statuses.append(response.status_code)
    summary = _load_summary(results)
    summary["sessions"] = session_count
    summary["turns"] = turns
    summary["delete_successes"] = sum(status == 200 for status in delete_statuses)
    return summary


async def _same_session_race(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_id: str,
    request_count: int,
) -> dict:
    create = await _post_chat(
        client,
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Start the session."}],
            "max_tokens": 2,
            "temperature": 0.0,
            "rwkv_context_mode": "stateful",
            "rwkv_session_id": session_id,
            "rwkv_session_action": "create",
        },
    )

    async def continue_session(index: int) -> RequestResult:
        return await _post_chat(
            client,
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Concurrent turn {index}."}],
                "max_tokens": 2,
                "temperature": 0.0,
                "rwkv_context_mode": "stateful",
                "rwkv_session_id": session_id,
                "rwkv_session_action": "continue",
            },
        )

    continuations = await asyncio.gather(
        *(continue_session(index) for index in range(request_count))
    )
    delete = await client.delete(f"{base_url}/v1/rwkv/sessions/{session_id}")
    return {
        "create_status": create.status_code,
        "continuation_summary": _load_summary(continuations),
        "delete_status": delete.status_code,
    }


async def _session_capacity_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session_count: int,
) -> dict:
    statuses = []
    created_session_ids = []
    prefix = f"capacity-{int(time.time())}"
    for index in range(session_count):
        session_id = f"{prefix}-{index}"
        result = await _post_chat(
            client,
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say ready."}],
                "max_tokens": 1,
                "temperature": 0.0,
                "rwkv_context_mode": "stateful",
                "rwkv_session_id": session_id,
                "rwkv_session_action": "create",
            },
        )
        statuses.append(result.status_code)
        if result.status_code == 200:
            created_session_ids.append(session_id)

    delete_statuses = []
    for session_id in created_session_ids:
        response = await client.delete(f"{base_url}/v1/rwkv/sessions/{session_id}")
        delete_statuses.append(response.status_code)
    return {
        "attempted": session_count,
        "statuses": statuses,
        "created": len(created_session_ids),
        "delete_successes": sum(status == 200 for status in delete_statuses),
    }


async def _main(args: argparse.Namespace) -> None:
    headers = _headers(args.api_key)
    limits = httpx.Limits(
        max_connections=max(args.concurrency, 8),
        max_keepalive_connections=max(args.concurrency, 8),
    )
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(
        headers=headers, limits=limits, timeout=timeout
    ) as client:
        result: dict = {}
        if args.mode in ("sliding", "all"):
            load_results = await _load_test(
                client,
                f"{args.base_url}/v1/chat/completions",
                args.model,
                args.requests,
                args.concurrency,
            )
            result["sliding_load"] = _load_summary(load_results)
            result["sliding_sample"] = load_results[0].body
        if args.mode in ("stateful", "all"):
            result["stateful_session"] = await _session_test(
                client, args.base_url, args.model, args.session_id
            )
        if args.tool_test:
            result["tool_calling"] = await _tool_test(client, args.base_url, args.model)
        if args.ttl_test:
            result["ttl"] = await _ttl_test(
                client,
                args.base_url,
                args.model,
                args.session_id,
                args.ttl_wait_s,
            )
        if args.stateful_tool_test:
            result["stateful_tool_calling"] = await _stateful_tool_test(
                client,
                args.base_url,
                args.model,
                f"{args.session_id}-tool",
            )
        if args.recovery_test:
            result["restart_recovery"] = await _recovery_test(
                client, args.base_url, args.model, args.session_id
            )
        if args.stateful_soak:
            result["stateful_soak"] = await _stateful_soak(
                client,
                args.base_url,
                args.model,
                args.soak_sessions,
                args.soak_turns,
                args.concurrency,
            )
        if args.same_session_race:
            result["same_session_race"] = await _same_session_race(
                client,
                args.base_url,
                args.model,
                f"{args.session_id}-race",
                args.race_requests,
            )
        if args.session_capacity_test:
            result["session_capacity"] = await _session_capacity_test(
                client,
                args.base_url,
                args.model,
                args.capacity_sessions,
            )
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("sliding", "stateful", "all"), default="all")
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--session-id", default="canary-session")
    parser.add_argument("--tool-test", action="store_true")
    parser.add_argument("--ttl-test", action="store_true")
    parser.add_argument("--ttl-wait-s", type=float, default=31.0)
    parser.add_argument("--stateful-tool-test", action="store_true")
    parser.add_argument("--recovery-test", action="store_true")
    parser.add_argument("--stateful-soak", action="store_true")
    parser.add_argument("--soak-sessions", type=int, default=8)
    parser.add_argument("--soak-turns", type=int, default=20)
    parser.add_argument("--same-session-race", action="store_true")
    parser.add_argument("--race-requests", type=int, default=8)
    parser.add_argument("--session-capacity-test", action="store_true")
    parser.add_argument("--capacity-sessions", type=int, default=9)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
