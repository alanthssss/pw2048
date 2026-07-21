#!/usr/bin/env python3
"""Generate valid inference traffic and assert release/error expectations."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


BOARD = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--expected-release")
    args = parser.parse_args()
    errors = 0
    releases: dict[str, int] = {}
    body = json.dumps({"board": BOARD}).encode()
    for i in range(args.requests):
        request = urllib.request.Request(
            f"{args.url.rstrip('/')}/v1/move", body,
            {"Content-Type": "application/json", "X-Request-ID": f"chaos-{i}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                release = response.headers.get("X-PW2048-Release", "unknown")
                releases[release] = releases.get(release, 0) + 1
        except (urllib.error.URLError, TimeoutError):
            errors += 1
    rate = errors / args.requests if args.requests else 1.0
    print(json.dumps({"requests": args.requests, "errors": errors,
                      "error_rate": rate, "releases": releases}, indent=2))
    if args.expected_release and set(releases) != {args.expected_release}:
        return 2
    return int(rate > args.max_error_rate)


if __name__ == "__main__":
    raise SystemExit(main())
