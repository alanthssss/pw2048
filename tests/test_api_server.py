import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from src.api_server import RateLimiter, ServiceSettings, make_handler


def _request(method, path, body=None, headers=None, settings=None):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler("greedy", "canary", "v2", settings or ServiceSettings()),
    )
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        payload = json.dumps(body) if body is not None else None
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, payload, request_headers)
        response = conn.getresponse()
        result = response.status, response.read().decode(), dict(response.headers)
        conn.close()
        return result
    finally:
        server.server_close()
        thread.join()


def test_health_and_release_header():
    status, body, headers = _request("GET", "/readyz")
    assert status == 200
    assert json.loads(body)["release"] == "canary"
    assert headers["X-PW2048-Release"] == "canary"


def test_inference_returns_versioned_result():
    board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    status, body, _ = _request("POST", "/v1/move", {"board": board})
    result = json.loads(body)
    assert status == 200
    assert result["move"] in {"up", "down", "left", "right"}
    assert result["release"] == "canary"
    assert result["model_version"] == "v2"


def test_invalid_board_is_rejected():
    status, body, _ = _request("POST", "/v1/move", {"board": [[2]]})
    assert status == 400
    assert "four rows" in json.loads(body)["error"]


def test_metrics_are_prometheus_compatible():
    status, body, _ = _request("GET", "/metrics")
    assert status == 200
    assert "pw2048_inference_requests_total" in body
    assert 'release="canary"' in body
    assert "pw2048_inference_rejected_total" in body
    assert "pw2048_inference_active" in body


def test_request_id_is_propagated():
    board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    status, body, headers = _request(
        "POST", "/v1/move", {"board": board}, {"X-Request-ID": "trace-123"}
    )
    assert status == 200
    assert headers["X-Request-ID"] == "trace-123"
    assert json.loads(body)["request_id"] == "trace-123"


def test_fault_injection_is_observable():
    board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    settings = ServiceSettings(fault_error_rate=1.0)
    status, body, _ = _request("POST", "/v1/move", {"board": board}, settings=settings)
    assert status == 500
    assert json.loads(body)["error"] == "internal server error"


def test_inference_timeout_returns_gateway_timeout():
    board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    settings = ServiceSettings(fault_delay_ms=5, inference_timeout_ms=1)
    status, body, _ = _request("POST", "/v1/move", {"board": board}, settings=settings)
    assert status == 504
    assert json.loads(body)["error"] == "inference timeout"


def test_rate_limiter_rejects_burst_over_limit():
    limiter = RateLimiter(1)
    assert limiter.allow()
    assert not limiter.allow()
