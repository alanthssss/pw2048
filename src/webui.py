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
from urllib.parse import parse_qs

# ---------------------------------------------------------------------------
# Constants (mirror main.py to avoid circular imports)
# ---------------------------------------------------------------------------

_DEFAULT_KEEP = 10
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1

# ---------------------------------------------------------------------------
# HTML pages (embedded so the module is fully self-contained)
# ---------------------------------------------------------------------------

_HTML_FORM = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pw2048 \u2013 Web Launcher</title>
  <style>
    :root {
      --accent: #5f9ea0;
      --accent-dark: #4a7e80;
      --bg: #1e1e2e;
      --surface: #2a2a3e;
      --border: #3a3a5c;
      --text: #e0e0f0;
      --muted: #9090a0;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 32px 16px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      width: 100%;
      max-width: 560px;
      padding: 32px;
    }
    h1 { font-size: 1.5rem; color: var(--accent); margin-bottom: 2px; }
    .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 24px; }
    .field { margin-bottom: 14px; }
    label {
      display: block;
      font-size: 0.8125rem;
      color: var(--muted);
      margin-bottom: 5px;
    }
    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      padding: 8px 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      font-size: 0.9375rem;
      outline: none;
      transition: border-color 0.15s;
    }
    input:focus, select:focus { border-color: var(--accent); }
    .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    .check-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .check-row input[type="checkbox"] {
      width: 16px; height: 16px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .check-row label { margin-bottom: 0; color: var(--text); cursor: pointer; }
    .mode-group { display: flex; gap: 8px; flex-wrap: wrap; }
    .mode-btn {
      padding: 5px 14px;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--bg);
      color: var(--text);
      cursor: pointer;
      font-size: 0.875rem;
      transition: all 0.15s;
    }
    .mode-btn:hover { border-color: var(--accent); }
    .mode-btn.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 600;
    }
    .section { border-top: 1px solid var(--border); margin-top: 20px; padding-top: 16px; }
    .section-title {
      font-size: 0.6875rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .btn-row { display: flex; justify-content: flex-end; gap: 12px; margin-top: 24px; }
    .btn {
      padding: 10px 22px;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      font-size: 0.9375rem;
      font-weight: 600;
      transition: background 0.15s;
    }
    .btn-secondary { background: var(--border); color: var(--text); }
    .btn-secondary:hover { background: #4a4a70; }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-dark); }
  </style>
</head>
<body>
<div class="card">
  <h1>pw2048</h1>
  <p class="subtitle">Web Launcher \u2014 configure your run and click Launch</p>

  <form method="post" action="/">

    <!-- Algorithm -->
    <div class="field">
      <label for="algorithm">Algorithm</label>
      <select id="algorithm" name="algorithm">
        <option value="random">random</option>
        <option value="greedy">greedy</option>
        <option value="heuristic">heuristic</option>
      </select>
    </div>

    <!-- Mode -->
    <div class="field">
      <label>Run mode</label>
      <div class="mode-group">
        <button type="button" class="mode-btn active" data-mode="custom"
                onclick="selectMode(this)">custom</button>
        <button type="button" class="mode-btn" data-mode="dev"
                onclick="selectMode(this)">dev</button>
        <button type="button" class="mode-btn" data-mode="release"
                onclick="selectMode(this)">release</button>
        <button type="button" class="mode-btn" data-mode="benchmark"
                onclick="selectMode(this)">benchmark</button>
      </div>
      <input type="hidden" name="mode" id="mode-input" value="custom">
    </div>

    <!-- Custom fields -->
    <div id="custom-fields">
      <div class="three-col">
        <div class="field">
          <label for="games">Games / run</label>
          <input type="number" id="games" name="games" value="20" min="1">
        </div>
        <div class="field">
          <label for="runs">Runs</label>
          <input type="number" id="runs" name="runs" value="1" min="1">
        </div>
        <div class="field">
          <label for="parallel">Workers</label>
          <input type="number" id="parallel" name="parallel" value="1" min="1">
        </div>
      </div>
    </div>

    <!-- Output -->
    <div class="section">
      <div class="section-title">Output</div>
      <div class="three-col">
        <div class="field" style="grid-column: span 2">
          <label for="output">Directory</label>
          <input type="text" id="output" name="output" value="results">
        </div>
        <div class="field">
          <label for="keep">Keep N runs</label>
          <input type="number" id="keep" name="keep" value="10" min="0">
        </div>
      </div>
    </div>

    <!-- Options -->
    <div class="section">
      <div class="section-title">Options</div>
      <div class="check-row">
        <input type="checkbox" id="show" name="show">
        <label for="show">Show browser window while playing</label>
      </div>
      <div class="check-row">
        <input type="checkbox" id="report" name="report">
        <label for="report">Generate HTML report after run</label>
      </div>
    </div>

    <!-- S3 -->
    <div class="section">
      <div class="section-title">S3 (optional)</div>
      <div class="field">
        <label for="s3_bucket">Bucket</label>
        <input type="text" id="s3_bucket" name="s3_bucket"
               placeholder="leave blank to skip">
      </div>
      <div class="field">
        <label for="s3_prefix">Key prefix</label>
        <input type="text" id="s3_prefix" name="s3_prefix" value="results">
      </div>
      <div class="check-row">
        <input type="checkbox" id="s3_public" name="s3_public">
        <label for="s3_public">Apply public-read ACL to uploaded objects</label>
      </div>
    </div>

    <div class="btn-row">
      <button type="reset" class="btn btn-secondary">Reset</button>
      <button type="submit" class="btn btn-primary">Launch \u25b6</button>
    </div>

  </form>
</div>

<script>
  function selectMode(btn) {
    document.querySelectorAll('.mode-btn').forEach(function(b) {
      b.classList.remove('active');
    });
    btn.classList.add('active');
    var mode = btn.getAttribute('data-mode');
    document.getElementById('mode-input').value = mode;
    document.getElementById('custom-fields').style.display =
      mode === 'custom' ? '' : 'none';
  }
</script>
</body>
</html>
"""

_HTML_SUCCESS = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>pw2048 \u2013 Launching\u2026</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #1e1e2e; color: #e0e0f0;
      display: flex; align-items: center; justify-content: center;
      height: 100vh; margin: 0;
    }
    .card { text-align: center; padding: 40px; }
    h1 { color: #5f9ea0; font-size: 2rem; margin-bottom: 12px; }
    p { color: #9090a0; margin-top: 8px; }
  </style>
</head>
<body>
<div class="card">
  <h1>\u2713 Launching\u2026</h1>
  <p>Configuration received.  Check your terminal for progress.</p>
  <p>You can close this tab.</p>
</div>
</body>
</html>
"""

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

    argv: list[str] = [
        "--algorithm", algorithm,
        "--output", output,
        "--keep", keep,
    ]

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

    bucket = _get("s3_bucket").strip()
    if bucket:
        argv += ["--s3-bucket", bucket, "--s3-prefix", _get("s3_prefix", "results")]
        if "s3_public" in form:
            argv.append("--s3-public")

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
