import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from src.api_server import make_handler


def _request(method, path, body=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("greedy", "canary", "v2"))
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, payload, {"Content-Type": "application/json"})
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
