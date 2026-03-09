"""Web UI launcher for pw2048.

Starts a local HTTP server on a random free port, opens the launcher form in
the system browser, and blocks until the user submits the form.  Returns an
argv list accepted by :func:`main.parse_args`.

No third-party packages are required — the implementation uses only the Python
standard library (``http.server``, ``threading``, ``webbrowser``,
``urllib.parse``).
"""

from __future__ import annotations

import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

# ---------------------------------------------------------------------------
# Constants (mirror main.py to avoid circular imports)
# ---------------------------------------------------------------------------

_DEFAULT_KEEP = 10
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1
_DEFAULT_EVAL_FREQ = 50
_DEFAULT_N_EVAL_GAMES = 20

# ---------------------------------------------------------------------------
# HTML pages — loaded from src/templates/ so styling lives in dedicated files
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_HTML_FORM: str = (_TEMPLATES_DIR / "webui_form.html").read_text(encoding="utf-8")
_HTML_SUCCESS: str = (_TEMPLATES_DIR / "webui_success.html").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Pure helper – testable without running the server
# ---------------------------------------------------------------------------

def _form_to_argv(form: dict[str, list[str]]) -> list[str]:
    """Convert a ``parse_qs``-style dict from the POST body to an argv list.

    Parameters
    ----------
    form:
        Mapping produced by :func:`urllib.parse.parse_qs` from the raw POST
        body.  Each key maps to a list of strings.

    Returns
    -------
    list[str]
        argv list suitable for :func:`main.parse_args`.
    """

    def _get(key: str, default: str = "") -> str:
        return form.get(key, [default])[0]

    algorithm = _get("algorithm", "random")
    mode_choice = _get("mode", "custom")
    output = _get("output", "results") or "results"
    keep = _get("keep", str(_DEFAULT_KEEP))
    algo_version = _get("algo_version", "").strip()

    argv: list[str] = [
        "--algorithm", algorithm,
        "--output", output,
        "--keep", keep,
    ]

    if algo_version:
        argv += ["--algo-version", algo_version]

    eval_freq = _get("eval_freq", str(_DEFAULT_EVAL_FREQ)).strip() or str(_DEFAULT_EVAL_FREQ)
    n_eval_games = _get("n_eval_games", str(_DEFAULT_N_EVAL_GAMES)).strip() or str(_DEFAULT_N_EVAL_GAMES)
    es_patience = _get("early_stopping_patience", "0").strip()

    train_games = _get("train_games", "0").strip()
    if train_games and int(train_games) > 0:
        argv += [
            "--train-games", train_games,
            "--eval-freq", eval_freq,
            "--n-eval-games", n_eval_games,
        ]
        tensorboard_dir = _get("tensorboard_dir", "").strip()
        if tensorboard_dir:
            argv += ["--tensorboard-dir", tensorboard_dir]

    if es_patience and int(es_patience) > 0:
        argv += [
            "--early-stopping-patience", es_patience,
            "--early-stopping-min-delta", _get("early_stopping_min_delta", "1").strip() or "1",
        ]
        # Only add eval-freq / n-eval-games if not already added by train_games block.
        if not (train_games and int(train_games) > 0):
            argv += ["--eval-freq", eval_freq, "--n-eval-games", n_eval_games]

    if mode_choice != "custom":
        argv += ["--mode", mode_choice]
    else:
        argv += [
            "--games", _get("games", str(_DEFAULT_GAMES)),
            "--runs", _get("runs", str(_DEFAULT_RUNS)),
            "--parallel", _get("parallel", str(_DEFAULT_PARALLEL)),
        ]

    if "show" in form:
        argv.append("--show")
    if "report" in form:
        argv.append("--report")

    return argv


# ---------------------------------------------------------------------------
# Web UI entry-point
# ---------------------------------------------------------------------------

def run_webui() -> list[str]:
    """Start a local web server, open the launcher form, and return argv.

    Opens ``http://127.0.0.1:<port>/`` in the system browser and blocks until
    the user submits the form.

    Returns
    -------
    list[str]
        Argument list that can be passed directly to :func:`main.parse_args`.
    """
    result_argv: list[list[str]] = []
    ready = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._send(200, "text/html; charset=utf-8", _HTML_FORM.encode())

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError:
                body = raw.decode("latin-1")
            form = parse_qs(body)
            result_argv[:] = _form_to_argv(form)
            self._send(200, "text/html; charset=utf-8", _HTML_SUCCESS.encode())
            ready.set()

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:  # suppress request log
            pass

    # Pick a free ephemeral port
    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        port = _s.getsockname()[1]

    server = HTTPServer(("127.0.0.1", port), _Handler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Web UI → {url}")
    print("  (fill in the form and click Launch — check your terminal for progress)\n")
    webbrowser.open(url)

    ready.wait()
    server.shutdown()
    return result_argv
