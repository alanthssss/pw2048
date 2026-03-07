"""Tests for src/gui.py – _build_argv helper and missing-tkinter behaviour."""

from __future__ import annotations

import builtins

import pytest

from src.gui import _build_argv

# ---------------------------------------------------------------------------
# Detect whether tkinter is actually installed in this environment.
# Used to decide whether the "no-tkinter" path is testable without mocking.
# ---------------------------------------------------------------------------
try:
    import tkinter as _tk
    _TKINTER_AVAILABLE = True
    del _tk
except ImportError:
    _TKINTER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _argv(**kwargs) -> list[str]:
    """Call _build_argv with sensible defaults, overriding only named keys."""
    defaults = dict(
        algorithm="random",
        mode_choice="custom",
        games="20",
        runs="1",
        parallel="1",
        output="results",
        show=False,
        keep="10",
        report=False,
        s3_bucket="",
        s3_prefix="results",
        s3_public=False,
    )
    defaults.update(kwargs)
    return _build_argv(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: _build_argv
# ---------------------------------------------------------------------------

class TestBuildArgvCustomMode:
    def test_returns_list(self):
        assert isinstance(_argv(), list)

    def test_algorithm_present(self):
        result = _argv(algorithm="greedy")
        assert "--algorithm" in result
        assert result[result.index("--algorithm") + 1] == "greedy"

    def test_games_present_in_custom_mode(self):
        result = _argv(games="50")
        assert "--games" in result
        assert result[result.index("--games") + 1] == "50"

    def test_runs_present_in_custom_mode(self):
        result = _argv(runs="3")
        assert "--runs" in result
        assert result[result.index("--runs") + 1] == "3"

    def test_parallel_present_in_custom_mode(self):
        result = _argv(parallel="4")
        assert "--parallel" in result
        assert result[result.index("--parallel") + 1] == "4"

    def test_output_present(self):
        result = _argv(output="out")
        assert "--output" in result
        assert result[result.index("--output") + 1] == "out"

    def test_empty_output_defaults_to_results(self):
        result = _argv(output="")
        assert result[result.index("--output") + 1] == "results"

    def test_keep_present(self):
        result = _argv(keep="5")
        assert "--keep" in result
        assert result[result.index("--keep") + 1] == "5"

    def test_no_mode_flag_for_custom(self):
        result = _argv(mode_choice="custom")
        assert "--mode" not in result

    def test_show_absent_by_default(self):
        assert "--show" not in _argv(show=False)

    def test_show_present_when_enabled(self):
        assert "--show" in _argv(show=True)

    def test_report_absent_by_default(self):
        assert "--report" not in _argv(report=False)

    def test_report_present_when_enabled(self):
        assert "--report" in _argv(report=True)

    def test_no_s3_flags_when_bucket_empty(self):
        result = _argv(s3_bucket="")
        assert "--s3-bucket" not in result
        assert "--s3-prefix" not in result

    def test_s3_flags_present_when_bucket_set(self):
        result = _argv(s3_bucket="my-bucket", s3_prefix="pfx")
        assert "--s3-bucket" in result
        assert result[result.index("--s3-bucket") + 1] == "my-bucket"
        assert "--s3-prefix" in result
        assert result[result.index("--s3-prefix") + 1] == "pfx"

    def test_s3_public_absent_by_default(self):
        result = _argv(s3_bucket="b", s3_public=False)
        assert "--s3-public" not in result

    def test_s3_public_present_when_set(self):
        result = _argv(s3_bucket="b", s3_public=True)
        assert "--s3-public" in result

    def test_s3_public_ignored_when_no_bucket(self):
        result = _argv(s3_bucket="", s3_public=True)
        assert "--s3-public" not in result

    def test_whitespace_bucket_treated_as_empty(self):
        result = _argv(s3_bucket="   ")
        assert "--s3-bucket" not in result


class TestBuildArgvPresetMode:
    def test_mode_flag_present(self):
        result = _argv(mode_choice="dev")
        assert "--mode" in result
        assert result[result.index("--mode") + 1] == "dev"

    def test_no_games_flag_for_preset(self):
        assert "--games" not in _argv(mode_choice="dev")

    def test_no_runs_flag_for_preset(self):
        assert "--runs" not in _argv(mode_choice="dev")

    def test_no_parallel_flag_for_preset(self):
        assert "--parallel" not in _argv(mode_choice="dev")

    def test_benchmark_mode(self):
        result = _argv(mode_choice="benchmark")
        assert result[result.index("--mode") + 1] == "benchmark"

    def test_release_mode(self):
        result = _argv(mode_choice="release")
        assert result[result.index("--mode") + 1] == "release"


class TestBuildArgvParseable:
    """Ensure _build_argv output is accepted by main.parse_args."""

    def test_custom_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(algorithm="heuristic", games="30", runs="2", parallel="2"))
        assert args.algorithm == "heuristic"
        assert args.games == 30
        assert args.runs == 2
        assert args.parallel == 2

    def test_preset_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(mode_choice="benchmark"))
        assert args.mode == "benchmark"
        assert args.games == 500

    def test_expectimax_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(algorithm="expectimax"))
        assert args.algorithm == "expectimax"

    def test_mcts_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(algorithm="mcts"))
        assert args.algorithm == "mcts"


# ---------------------------------------------------------------------------
# Tests: run_gui() when tkinter is absent
# ---------------------------------------------------------------------------

class TestRunGuiMissingTkinter:
    """run_gui() raises SystemExit with installation instructions when tkinter
    cannot be imported."""

    def test_raises_systemexit(self, monkeypatch):
        real_import = builtins.__import__

        def _block_tkinter(name, *args, **kwargs):
            if name in ("tkinter", "tkinter.ttk", "tkinter.filedialog"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_tkinter)

        from src.gui import run_gui

        with pytest.raises(SystemExit):
            run_gui()

    @pytest.mark.skipif(
        _TKINTER_AVAILABLE,
        reason="tkinter is installed; the missing-tkinter path cannot be tested",
    )
    def test_raises_systemexit_naturally(self):
        """In environments where tkinter is genuinely absent, no mocking needed."""
        from src.gui import run_gui

        with pytest.raises(SystemExit):
            run_gui()
