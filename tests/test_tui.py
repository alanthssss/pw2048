"""Tests for src/tui.py – the interactive TUI wizard."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.tui import run_tui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_answers(**overrides):
    """Return a mapping of questionary prompt → answer used in run_tui().

    The defaults represent a minimal "custom mode" run.  Pass keyword
    arguments to override individual answers.
    """
    defaults = {
        "algorithm": "random",
        "mode_choice": "custom",
        "games": "20",
        "runs": "1",
        "parallel": "1",
        "output": "results",
        "show": False,
        "keep_str": "10",
        "report": False,
        "use_s3": False,
        "proceed": True,
    }
    defaults.update(overrides)
    return defaults


def _patch_tui(answers: dict):
    """Context manager that patches all questionary calls used by run_tui().

    Returns a context manager whose ``__enter__`` yields nothing (the
    patching happens as a side-effect).
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        call_log: list[str] = []

        def _select_side_effect(*args, **kwargs):
            m = MagicMock()
            # Determine which select call this is by order of call_log
            if "Algorithm:" in args[0]:
                m.ask.return_value = answers["algorithm"]
            elif "Run mode:" in args[0]:
                m.ask.return_value = answers["mode_choice"]
            else:
                m.ask.return_value = None
            return m

        def _text_side_effect(*args, **kwargs):
            m = MagicMock()
            prompt = args[0] if args else kwargs.get("message", "")
            prompt_lower = prompt.lower()
            if "games" in prompt_lower:
                m.ask.return_value = answers["games"]
            elif "number of runs" in prompt_lower:
                m.ask.return_value = answers["runs"]
            elif "parallel" in prompt_lower:
                m.ask.return_value = answers["parallel"]
            elif "output" in prompt_lower:
                m.ask.return_value = answers["output"]
            elif "keep" in prompt_lower:
                m.ask.return_value = answers["keep_str"]
            elif "bucket" in prompt_lower:
                m.ask.return_value = answers.get("s3_bucket", "my-bucket")
            elif "prefix" in prompt_lower:
                m.ask.return_value = answers.get("s3_prefix", "results")
            else:
                m.ask.return_value = answers.get("output", "results")
            return m

        def _confirm_side_effect(*args, **kwargs):
            m = MagicMock()
            prompt = args[0] if args else kwargs.get("message", "")
            if "browser" in prompt.lower():
                m.ask.return_value = answers["show"]
            elif "report" in prompt.lower():
                m.ask.return_value = answers["report"]
            elif "s3" in prompt.lower():
                m.ask.return_value = answers["use_s3"]
            elif "public" in prompt.lower():
                m.ask.return_value = answers.get("s3_public", False)
            elif "proceed" in prompt.lower():
                m.ask.return_value = answers["proceed"]
            else:
                m.ask.return_value = False
            return m

        with (
            patch("src.tui.questionary.select", side_effect=_select_side_effect),
            patch("src.tui.questionary.text", side_effect=_text_side_effect),
            patch("src.tui.questionary.confirm", side_effect=_confirm_side_effect),
        ):
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Tests: argv structure for different configurations
# ---------------------------------------------------------------------------


class TestRunTuiCustomMode:
    def test_returns_list(self):
        with _patch_tui(_mock_answers()):
            result = run_tui()
        assert isinstance(result, list)

    def test_algorithm_present(self):
        with _patch_tui(_mock_answers(algorithm="greedy")):
            result = run_tui()
        assert "--algorithm" in result
        assert result[result.index("--algorithm") + 1] == "greedy"

    def test_custom_games_present(self):
        with _patch_tui(_mock_answers(games="50")):
            result = run_tui()
        assert "--games" in result
        assert result[result.index("--games") + 1] == "50"

    def test_custom_runs_present(self):
        with _patch_tui(_mock_answers(runs="3")):
            result = run_tui()
        assert "--runs" in result
        assert result[result.index("--runs") + 1] == "3"

    def test_custom_parallel_present(self):
        with _patch_tui(_mock_answers(parallel="4")):
            result = run_tui()
        assert "--parallel" in result
        assert result[result.index("--parallel") + 1] == "4"

    def test_output_present(self):
        with _patch_tui(_mock_answers(output="out")):
            result = run_tui()
        assert "--output" in result
        assert result[result.index("--output") + 1] == "out"

    def test_keep_present(self):
        with _patch_tui(_mock_answers(keep_str="5")):
            result = run_tui()
        assert "--keep" in result
        assert result[result.index("--keep") + 1] == "5"

    def test_show_flag_absent_by_default(self):
        with _patch_tui(_mock_answers(show=False)):
            result = run_tui()
        assert "--show" not in result

    def test_show_flag_present_when_enabled(self):
        with _patch_tui(_mock_answers(show=True)):
            result = run_tui()
        assert "--show" in result

    def test_report_flag_absent_by_default(self):
        with _patch_tui(_mock_answers(report=False)):
            result = run_tui()
        assert "--report" not in result

    def test_report_flag_present_when_enabled(self):
        with _patch_tui(_mock_answers(report=True)):
            result = run_tui()
        assert "--report" in result

    def test_no_mode_flag_for_custom(self):
        with _patch_tui(_mock_answers(mode_choice="custom")):
            result = run_tui()
        assert "--mode" not in result

    def test_no_s3_flags_by_default(self):
        with _patch_tui(_mock_answers(use_s3=False)):
            result = run_tui()
        assert "--s3-bucket" not in result


class TestRunTuiPresetMode:
    def test_dev_mode_uses_mode_flag(self):
        with _patch_tui(_mock_answers(mode_choice="dev")):
            result = run_tui()
        assert "--mode" in result
        assert result[result.index("--mode") + 1] == "dev"

    def test_preset_mode_has_no_games_flag(self):
        with _patch_tui(_mock_answers(mode_choice="dev")):
            result = run_tui()
        assert "--games" not in result

    def test_preset_mode_has_no_runs_flag(self):
        with _patch_tui(_mock_answers(mode_choice="dev")):
            result = run_tui()
        assert "--runs" not in result

    def test_preset_mode_has_no_parallel_flag(self):
        with _patch_tui(_mock_answers(mode_choice="dev")):
            result = run_tui()
        assert "--parallel" not in result

    def test_benchmark_mode(self):
        with _patch_tui(_mock_answers(mode_choice="benchmark")):
            result = run_tui()
        assert "--mode" in result
        assert result[result.index("--mode") + 1] == "benchmark"


class TestRunTuiS3:
    def test_s3_bucket_flag_present_when_enabled(self):
        with _patch_tui(
            _mock_answers(use_s3=True, s3_bucket="my-bucket", s3_prefix="results")
        ):
            result = run_tui()
        assert "--s3-bucket" in result
        assert result[result.index("--s3-bucket") + 1] == "my-bucket"

    def test_s3_prefix_flag_present_when_enabled(self):
        with _patch_tui(
            _mock_answers(use_s3=True, s3_bucket="my-bucket", s3_prefix="pfx")
        ):
            result = run_tui()
        assert "--s3-prefix" in result
        assert result[result.index("--s3-prefix") + 1] == "pfx"

    def test_s3_public_flag_absent_by_default(self):
        with _patch_tui(
            _mock_answers(use_s3=True, s3_bucket="b", s3_public=False)
        ):
            result = run_tui()
        assert "--s3-public" not in result

    def test_s3_public_flag_present_when_set(self):
        with _patch_tui(
            _mock_answers(use_s3=True, s3_bucket="b", s3_public=True)
        ):
            result = run_tui()
        assert "--s3-public" in result


class TestRunTuiCancellation:
    def test_cancel_at_algorithm_raises_systemexit(self):
        with _patch_tui(_mock_answers(algorithm=None)):
            with pytest.raises(SystemExit):
                run_tui()

    def test_cancel_at_proceed_raises_systemexit(self):
        with _patch_tui(_mock_answers(proceed=False)):
            with pytest.raises(SystemExit):
                run_tui()


class TestRunTuiArgvParseable:
    """Ensure the argv produced by run_tui() is accepted by parse_args()."""

    def test_custom_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers()):
            argv = run_tui()
        args = parse_args(argv)
        assert args.algorithm == "random"
        assert args.games == 20
        assert args.runs == 1
        assert args.parallel == 1

    def test_preset_mode_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers(mode_choice="dev")):
            argv = run_tui()
        args = parse_args(argv)
        assert args.mode == "dev"
        assert args.games == 100

    def test_expectimax_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers(algorithm="expectimax")):
            argv = run_tui()
        args = parse_args(argv)
        assert args.algorithm == "expectimax"

    def test_mcts_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers(algorithm="mcts")):
            argv = run_tui()
        args = parse_args(argv)
        assert args.algorithm == "mcts"

    def test_dqn_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers(algorithm="dqn")):
            argv = run_tui()
        args = parse_args(argv)
        assert args.algorithm == "dqn"

    def test_ppo_argv_is_valid(self):
        from main import parse_args

        with _patch_tui(_mock_answers(algorithm="ppo")):
            argv = run_tui()
        args = parse_args(argv)
        assert args.algorithm == "ppo"
