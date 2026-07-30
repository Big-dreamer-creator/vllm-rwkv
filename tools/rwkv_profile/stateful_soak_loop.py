#!/usr/bin/env python3
"""Run repeated stateful canary cycles for a bounded soak window."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="rwkv-stateful")
    parser.add_argument("--duration-s", type=float, default=24 * 60 * 60)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--soak-sessions", type=int, default=8)
    parser.add_argument("--soak-turns", type=int, default=20)
    parser.add_argument("--race-requests", type=int, default=8)
    parser.add_argument("--capacity-sessions", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canary = Path(__file__).with_name("openai_canary.py")
    command = [
        sys.executable,
        str(canary),
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--mode",
        "all",
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--tool-test",
        "--stateful-tool-test",
        "--stateful-soak",
        "--soak-sessions",
        str(args.soak_sessions),
        "--soak-turns",
        str(args.soak_turns),
        "--same-session-race",
        "--race-requests",
        str(args.race_requests),
        "--session-capacity-test",
        "--capacity-sessions",
        str(args.capacity_sessions),
        "--timeout-s",
        str(args.timeout_s),
    ]
    deadline = time.monotonic() + args.duration_s
    cycle = 0

    while time.monotonic() < deadline:
        cycle += 1
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        print(f"soak_cycle_start cycle={cycle} at={started}", flush=True)
        cycle_start = time.monotonic()
        result = subprocess.run(command, check=False)
        elapsed = time.monotonic() - cycle_start
        print(
            f"soak_cycle_end cycle={cycle} returncode={result.returncode} "
            f"elapsed_s={elapsed:.3f}",
            flush=True,
        )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(args.interval_s, remaining))

    print(f"soak_finished cycles={cycle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
