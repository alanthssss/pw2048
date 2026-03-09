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

    def test_no_s3_flags_emitted(self):
        result = _argv()
        assert "--s3-bucket" not in result
        assert "--s3-public" not in result


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

    def test_dqn_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(algorithm="dqn"))
        assert args.algorithm == "dqn"

    def test_ppo_argv_valid(self):
        from main import parse_args

        args = parse_args(_argv(algorithm="ppo"))
        assert args.algorithm == "ppo"


# ---------------------------------------------------------------------------
# Tests: RL training flags in _build_argv
# ---------------------------------------------------------------------------

class TestBuildArgvRLTraining:
    """New RL training flags: --train-games, --eval-freq, --n-eval-games,
    --tensorboard-dir."""

    def test_train_games_absent_when_zero(self):
        result = _argv(train_games="0")
        assert "--train-games" not in result

    def test_train_games_present_when_positive(self):
        result = _argv(train_games="5000")
        assert "--train-games" in result
        assert result[result.index("--train-games") + 1] == "5000"

    def test_eval_freq_included_with_train_games(self):
        result = _argv(train_games="100", eval_freq="25")
        assert "--eval-freq" in result
        assert result[result.index("--eval-freq") + 1] == "25"

    def test_n_eval_games_included_with_train_games(self):
        result = _argv(train_games="100", n_eval_games="10")
        assert "--n-eval-games" in result
        assert result[result.index("--n-eval-games") + 1] == "10"

    def test_tensorboard_dir_included_when_set(self):
        result = _argv(train_games="100", tensorboard_dir="tb_logs")
        assert "--tensorboard-dir" in result
        assert result[result.index("--tensorboard-dir") + 1] == "tb_logs"

    def test_tensorboard_dir_absent_when_empty(self):
        result = _argv(train_games="100", tensorboard_dir="")
        assert "--tensorboard-dir" not in result

    def test_rl_flags_absent_when_train_games_zero(self):
        result = _argv(train_games="0", tensorboard_dir="tb_logs")
        assert "--train-games" not in result
        assert "--eval-freq" not in result
        assert "--n-eval-games" not in result
        assert "--tensorboard-dir" not in result

    def test_rl_argv_parseable_by_main(self):
        from main import parse_args

        result = _argv(
            algorithm="dqn",
            train_games="200",
            eval_freq="50",
            n_eval_games="10",
            tensorboard_dir="tb_logs",
        )
        args = parse_args(result)
        assert args.train_games == 200
        assert args.eval_freq == 50
        assert args.n_eval_games == 10
        assert args.tensorboard_dir == "tb_logs"
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
