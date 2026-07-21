"""Production-shaped HTTP inference service for pw2048.

The service intentionally uses only the Python standard library so the same
container can be promoted from development to production without changing its
runtime dependencies.  A deployment selects its model through environment
variables; the response and metrics expose immutable release metadata.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.algorithms.expectimax_algo import ExpectimaxAlgorithm
from src.algorithms.greedy_algo import GreedyAlgorithm
from src.algorithms.heuristic_algo import HeuristicAlgorithm


ALGORITHMS = {
    "greedy": GreedyAlgorithm,
    "heuristic": HeuristicAlgorithm,
    "expectimax": ExpectimaxAlgorithm,
}
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.duration_sum = 0.0
        self.buckets = {b: 0 for b in BUCKETS}

    def observe(self, duration: float, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.errors += int(error)
            self.duration_sum += duration
            for bucket in BUCKETS:
                if duration <= bucket:
                    self.buckets[bucket] += 1

    def render(self, release: str, model: str) -> bytes:
        labels = f'release="{release}",model="{model}"'
        with self._lock:
            lines = [
                "# HELP pw2048_inference_requests_total Inference requests.",
                "# TYPE pw2048_inference_requests_total counter",
                f"pw2048_inference_requests_total{{{labels}}} {self.requests}",
                "# HELP pw2048_inference_errors_total Failed inference requests.",
                "# TYPE pw2048_inference_errors_total counter",
                f"pw2048_inference_errors_total{{{labels}}} {self.errors}",
                "# HELP pw2048_inference_duration_seconds Inference latency.",
                "# TYPE pw2048_inference_duration_seconds histogram",
            ]
            for bucket, count in self.buckets.items():
                lines.append(
                    f'pw2048_inference_duration_seconds_bucket{{{labels},le="{bucket}"}} {count}'
                )
            lines.extend([
                f'pw2048_inference_duration_seconds_bucket{{{labels},le="+Inf"}} {self.requests}',
                f"pw2048_inference_duration_seconds_sum{{{labels}}} {self.duration_sum:.9f}",
                f"pw2048_inference_duration_seconds_count{{{labels}}} {self.requests}",
                "# HELP pw2048_build_info Immutable release metadata.",
                "# TYPE pw2048_build_info gauge",
                f'pw2048_build_info{{{labels},git_sha="{os.getenv("GIT_SHA", "unknown")}"}} 1',
                "",
            ])
        return "\n".join(lines).encode()


def _validate_board(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("board must contain exactly four rows")
    board: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("each board row must contain exactly four cells")
        if any(not isinstance(cell, int) or cell < 0 for cell in row):
            raise ValueError("board cells must be non-negative integers")
        board.append(row)
    return board


def make_handler(algorithm_name: str, release: str, model_version: str):
    metrics = Metrics()
    algorithm_class = ALGORITHMS[algorithm_name]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/healthz", "/readyz"):
                self._json(200, {"status": "ok", "release": release})
            elif self.path == "/version":
                self._json(200, {
                    "release": release,
                    "model": algorithm_name,
                    "model_version": model_version,
                    "git_sha": os.getenv("GIT_SHA", "unknown"),
                })
            elif self.path == "/metrics":
                body = metrics.render(release, f"{algorithm_name}:{model_version}")
                self._send(200, "text/plain; version=0.0.4", body)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/move":
                self._json(404, {"error": "not found"})
                return
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64 * 1024:
                    raise ValueError("request body must be between 1 byte and 64 KiB")
                payload = json.loads(self.rfile.read(length))
                board = _validate_board(payload.get("board"))
                move = algorithm_class().choose_move(board)
                metrics.observe(time.perf_counter() - started)
                self._json(200, {
                    "move": move,
                    "release": release,
                    "model": algorithm_name,
                    "model_version": model_version,
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                metrics.observe(time.perf_counter() - started, error=True)
                self._json(400, {"error": str(exc), "release": release})
            except Exception:
                metrics.observe(time.perf_counter() - started, error=True)
                self._json(500, {"error": "internal server error", "release": release})

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, "application/json", json.dumps(payload).encode())

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-PW2048-Release", release)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            print(json.dumps({
                "time": time.time(), "release": release,
                "client": self.client_address[0], "message": fmt % args,
            }), flush=True)

    return Handler


def main() -> None:
    algorithm = os.getenv("MODEL_NAME", "greedy").lower()
    if algorithm not in ALGORITHMS:
        raise SystemExit(f"MODEL_NAME must be one of: {', '.join(ALGORITHMS)}")
    release = os.getenv("RELEASE_TRACK", "local")
    model_version = os.getenv("MODEL_VERSION", "v1")
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(algorithm, release, model_version))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(json.dumps({"event": "ready", "port": port, "release": release,
                      "model": algorithm, "model_version": model_version}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
