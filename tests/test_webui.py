"""Tests for src/webui.py – _form_to_argv helper and server behaviour."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

import pytest

from src.webui import _form_to_argv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _form(**kwargs) -> dict[str, list[str]]:
    """Build a parse_qs-style dict with sensible defaults."""
    defaults: dict[str, list[str]] = {
        "algorithm": ["random"],
        "mode": ["custom"],
        "games": ["20"],
        "runs": ["1"],
        "parallel": ["1"],
        "output": ["results"],
        "keep": ["10"],
    }
    for k, v in kwargs.items():
        if v is None:
            defaults.pop(k, None)
        else:
            defaults[k] = [v] if isinstance(v, str) else v
    return defaults


# ---------------------------------------------------------------------------
# Tests: _form_to_argv
# ---------------------------------------------------------------------------


class TestFormToArgvCustomMode:
    def test_returns_list(self):
        assert isinstance(_form_to_argv(_form()), list)

    def test_algorithm_present(self):
        result = _form_to_argv(_form(algorithm="greedy"))
        assert "--algorithm" in result
        assert result[result.index("--algorithm") + 1] == "greedy"

    def test_games_present(self):
        result = _form_to_argv(_form(games="50"))
        assert "--games" in result
        assert result[result.index("--games") + 1] == "50"

    def test_runs_present(self):
        result = _form_to_argv(_form(runs="3"))
        assert "--runs" in result
        assert result[result.index("--runs") + 1] == "3"

    def test_parallel_present(self):
        result = _form_to_argv(_form(parallel="4"))
        assert "--parallel" in result
        assert result[result.index("--parallel") + 1] == "4"

    def test_output_present(self):
        result = _form_to_argv(_form(output="out"))
        assert "--output" in result
        assert result[result.index("--output") + 1] == "out"

    def test_empty_output_defaults_to_results(self):
        result = _form_to_argv(_form(output=""))
        assert result[result.index("--output") + 1] == "results"

    def test_keep_present(self):
        result = _form_to_argv(_form(keep="5"))
        assert "--keep" in result
        assert result[result.index("--keep") + 1] == "5"

    def test_no_mode_flag_for_custom(self):
        result = _form_to_argv(_form(mode="custom"))
        assert "--mode" not in result

    def test_show_absent_by_default(self):
        result = _form_to_argv(_form())
        assert "--show" not in result

    def test_show_present_when_checkbox_checked(self):
        result = _form_to_argv(_form(show="on"))
        assert "--show" in result

    def test_report_absent_by_default(self):
        result = _form_to_argv(_form())
        assert "--report" not in result

    def test_report_present_when_checkbox_checked(self):
        result = _form_to_argv(_form(report="on"))
        assert "--report" in result

    def test_no_s3_flags_when_bucket_empty(self):
        result = _form_to_argv(_form(s3_bucket=""))
        assert "--s3-bucket" not in result
        assert "--s3-prefix" not in result

    def test_s3_flags_present_when_bucket_set(self):
        result = _form_to_argv(_form(s3_bucket="my-bucket", s3_prefix="pfx"))
        assert "--s3-bucket" in result
        assert result[result.index("--s3-bucket") + 1] == "my-bucket"
        assert "--s3-prefix" in result
        assert result[result.index("--s3-prefix") + 1] == "pfx"

    def test_s3_public_absent_by_default(self):
        result = _form_to_argv(_form(s3_bucket="b"))
        assert "--s3-public" not in result

    def test_s3_public_present_when_checked(self):
        result = _form_to_argv(_form(s3_bucket="b", s3_prefix="results", s3_public="on"))
        assert "--s3-public" in result

    def test_s3_public_ignored_when_no_bucket(self):
        result = _form_to_argv(_form(s3_bucket="", s3_public="on"))
        assert "--s3-public" not in result

    def test_whitespace_bucket_treated_as_empty(self):
        result = _form_to_argv(_form(s3_bucket="   "))
        assert "--s3-bucket" not in result


class TestFormToArgvPresetMode:
    def test_mode_flag_present(self):
        result = _form_to_argv(_form(mode="dev"))
        assert "--mode" in result
        assert result[result.index("--mode") + 1] == "dev"

    def test_no_games_flag_for_preset(self):
        assert "--games" not in _form_to_argv(_form(mode="dev"))

    def test_no_runs_flag_for_preset(self):
        assert "--runs" not in _form_to_argv(_form(mode="dev"))

    def test_no_parallel_flag_for_preset(self):
        assert "--parallel" not in _form_to_argv(_form(mode="dev"))

    def test_benchmark_mode(self):
        result = _form_to_argv(_form(mode="benchmark"))
        assert result[result.index("--mode") + 1] == "benchmark"

    def test_release_mode(self):
        result = _form_to_argv(_form(mode="release"))
        assert result[result.index("--mode") + 1] == "release"


class TestFormToArgvParseable:
    """Ensure _form_to_argv output is accepted by main.parse_args."""

    def test_custom_argv_valid(self):
        from main import parse_args

        form = _form(algorithm="heuristic", games="30", runs="2", parallel="2", report="on")
        args = parse_args(_form_to_argv(form))
        assert args.algorithm == "heuristic"
        assert args.games == 30
        assert args.runs == 2
        assert args.parallel == 2
        assert args.report is True

    def test_preset_argv_valid(self):
        from main import parse_args

        args = parse_args(_form_to_argv(_form(mode="benchmark")))
        assert args.mode == "benchmark"
        assert args.games == 500

    def test_expectimax_argv_valid(self):
        from main import parse_args

        args = parse_args(_form_to_argv(_form(algorithm="expectimax")))
        assert args.algorithm == "expectimax"

    def test_mcts_argv_valid(self):
        from main import parse_args

        args = parse_args(_form_to_argv(_form(algorithm="mcts")))
        assert args.algorithm == "mcts"

    def test_dqn_argv_valid(self):
        from main import parse_args

        args = parse_args(_form_to_argv(_form(algorithm="dqn")))
        assert args.algorithm == "dqn"

    def test_ppo_argv_valid(self):
        from main import parse_args

        args = parse_args(_form_to_argv(_form(algorithm="ppo")))
        assert args.algorithm == "ppo"


# ---------------------------------------------------------------------------
# Tests: HTTP server integration
# ---------------------------------------------------------------------------


class TestWebUIServer:
    """Start the HTTP server directly (without opening a browser) and verify
    the GET/POST responses."""

    @staticmethod
    def _start_server():
        """Spin up the internal server handler on a free port and return
        (server, port).  Caller is responsible for calling server.shutdown()."""
        import socket
        import threading
        from http.server import HTTPServer
        from src.webui import _form_to_argv, _HTML_FORM, _HTML_SUCCESS
        from urllib.parse import parse_qs

        result_argv: list[list[str]] = []
        ready = threading.Event()

        class TestHandler(  # type: ignore[misc]
            __import__("http.server", fromlist=["BaseHTTPRequestHandler"]).BaseHTTPRequestHandler
        ):
            def do_GET(self) -> None:
                body = _HTML_FORM.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(length).decode()
                form = parse_qs(body_bytes)
                result_argv[:] = _form_to_argv(form)
                resp = _HTML_SUCCESS.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                ready.set()

            def log_message(self, *_args: object) -> None:
                pass

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server = HTTPServer(("127.0.0.1", port), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port, result_argv, ready

    def test_get_returns_html_form(self):
        server, port, _argv_out, _ready = self._start_server()
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            body = resp.read().decode()
            assert "pw2048" in body
            assert "<form" in body
        finally:
            server.shutdown()

    def test_post_returns_success_page(self):
        server, port, _argv_out, _ready = self._start_server()
        try:
            data = urllib.parse.urlencode(
                {"algorithm": "greedy", "mode": "custom",
                 "games": "10", "runs": "1", "parallel": "1",
                 "output": "results", "keep": "10"}
            ).encode()
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", data=data
            )
            body = resp.read().decode()
            assert "Launching" in body
        finally:
            server.shutdown()

    def test_post_produces_correct_argv(self):
        server, port, result_argv, ready = self._start_server()
        try:
            data = urllib.parse.urlencode(
                {"algorithm": "greedy", "mode": "custom",
                 "games": "25", "runs": "2", "parallel": "1",
                 "output": "myresults", "keep": "5", "report": "on"}
            ).encode()
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", data=data)
            ready.wait(timeout=5)
        finally:
            server.shutdown()

        assert "--algorithm" in result_argv
        assert result_argv[result_argv.index("--algorithm") + 1] == "greedy"
        assert "--games" in result_argv
        assert result_argv[result_argv.index("--games") + 1] == "25"
        assert "--report" in result_argv

    def test_html_form_contains_mode_buttons(self):
        from src.webui import _HTML_FORM

        for mode in ("custom", "dev", "release", "benchmark"):
            assert mode in _HTML_FORM

    def test_html_form_contains_algorithm_options(self):
        from src.webui import _HTML_FORM

        for algo in ("random", "greedy", "heuristic", "expectimax", "mcts", "dqn", "ppo"):
            assert algo in _HTML_FORM

    def test_html_form_uses_2048_palette(self):
        """The launch form should use the 2048-inspired palette, not the old dark theme."""
        from src.webui import _HTML_FORM

        # Must contain the 2048 tan background colour
        assert "#faf8ef" in _HTML_FORM
        # Must contain the 2048 brownish-tan header colour
        assert "#bbada0" in _HTML_FORM
        # Must contain the 2048 orange accent colour
        assert "#f59563" in _HTML_FORM
        # Must NOT use the old dark purple background
        assert "#1e1e2e" not in _HTML_FORM

    def test_html_success_uses_2048_palette(self):
        """The success page should use the 2048-inspired palette."""
        from src.webui import _HTML_SUCCESS

        assert "#faf8ef" in _HTML_SUCCESS
        assert "#bbada0" in _HTML_SUCCESS
        assert "#1e1e2e" not in _HTML_SUCCESS

