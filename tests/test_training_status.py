"""Tests for src/training_status.py and the --inspect-checkpoint / --training-status CLI flags."""

from __future__ import annotations

import csv
import pathlib
import tempfile

import numpy as np
import pytest

from src.training_status import inspect_checkpoint, print_training_status


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dqn_checkpoint(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal DQN-style .npz checkpoint file."""
    ckpt = tmp_path / "checkpoint.npz"
    rng = np.random.default_rng(0)
    # Minimal DQN-v3 checkpoint (two-layer net, matches save_checkpoint format).
    np.savez(
        ckpt,
        q_W1=rng.normal(size=(256, 256)).astype(np.float32),
        q_b1=rng.normal(size=(256,)).astype(np.float32),
        q_W2=rng.normal(size=(256, 256)).astype(np.float32),
        q_b2=rng.normal(size=(256,)).astype(np.float32),
        q_W3=rng.normal(size=(256, 4)).astype(np.float32),
        q_b3=rng.normal(size=(4,)).astype(np.float32),
        t_W1=rng.normal(size=(256, 256)).astype(np.float32),
        t_b1=rng.normal(size=(256,)).astype(np.float32),
        t_W2=rng.normal(size=(256, 256)).astype(np.float32),
        t_b2=rng.normal(size=(256,)).astype(np.float32),
        t_W3=rng.normal(size=(256, 4)).astype(np.float32),
        t_b3=rng.normal(size=(4,)).astype(np.float32),
        epsilon=np.array([0.15], dtype=np.float32),
        step=np.array([12345], dtype=np.int64),
        adam_t=np.array([500], dtype=np.int64),
    )
    return ckpt


@pytest.fixture()
def training_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a synthetic training_log.csv."""
    rows = [
        ("step", "tag", "value"),
    ]
    # 100 training games
    for g in range(1, 101):
        rows.append((g, "train/score", float(1000 + g * 10)))
        rows.append((g, "train/max_tile", 128.0))
        rows.append((g, "train/epsilon", max(0.05, 0.9 - g * 0.008)))
        if g % 10 == 0:
            rows.append((g, "eval/mean_score", float(800 + g * 12)))
            rows.append((g, "eval/max_score", float(1200 + g * 15)))
            rows.append((g, "eval/mean_tile", 128.0))
            rows.append((g, "eval/max_tile", 256.0))
    csv_path = tmp_path / "training_log.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    return tmp_path   # return the directory, not the file


# ---------------------------------------------------------------------------
# inspect_checkpoint
# ---------------------------------------------------------------------------

class TestInspectCheckpoint:
    def test_returns_dict_with_expected_keys(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        for key in ("algo", "step", "epsilon", "file_size_kb", "n_params", "weight_norms", "adam_t"):
            assert key in result, f"Missing key: {key}"

    def test_detects_dqn_algorithm(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        assert result["algo"] == "DQN-v3"

    def test_step_and_epsilon_values(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        assert result["step"] == 12345
        assert abs(result["epsilon"] - 0.15) < 1e-5

    def test_adam_t_value(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        assert result["adam_t"] == 500

    def test_n_params_positive(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        assert result["n_params"] > 0

    def test_weight_norms_all_positive(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        assert all(v > 0 for v in result["weight_norms"].values())

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_checkpoint(tmp_path / "nonexistent.npz", verbose=False)

    def test_verbose_prints(self, dqn_checkpoint, capsys):
        inspect_checkpoint(dqn_checkpoint, verbose=True)
        captured = capsys.readouterr()
        assert "DQN-v3" in captured.out
        assert "epsilon" in captured.out.lower() or "ε" in captured.out

    def test_file_size_kb_correct(self, dqn_checkpoint):
        result = inspect_checkpoint(dqn_checkpoint, verbose=False)
        expected_kb = dqn_checkpoint.stat().st_size / 1024
        assert abs(result["file_size_kb"] - expected_kb) < 0.5


# ---------------------------------------------------------------------------
# print_training_status
# ---------------------------------------------------------------------------

class TestPrintTrainingStatus:
    def test_returns_dict_with_expected_keys(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        for key in ("total_train_games", "best_eval_score", "recent_eval_mean",
                    "trend", "stable", "eval_rounds", "current_epsilon"):
            assert key in result, f"Missing key: {key}"

    def test_total_train_games(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        assert result["total_train_games"] == 100

    def test_eval_rounds(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        assert result["eval_rounds"] == 10  # every 10 games

    def test_best_eval_score_positive(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        assert result["best_eval_score"] > 0

    def test_trend_is_numeric(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        import math
        assert not math.isnan(result["trend"])

    def test_current_epsilon_in_range(self, training_csv):
        result = print_training_status(training_csv, verbose=False)
        # Epsilon was capped at 0.05 in our fixture.
        assert 0.0 <= result["current_epsilon"] <= 1.0

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            print_training_status(tmp_path / "nonexistent_dir", verbose=False)

    def test_verbose_prints(self, training_csv, capsys):
        print_training_status(training_csv, verbose=True)
        captured = capsys.readouterr()
        assert "Train games" in captured.out
        assert "Best eval" in captured.out

    def test_sparkline_printed_when_verbose(self, training_csv, capsys):
        print_training_status(training_csv, verbose=True)
        captured = capsys.readouterr()
        assert "sparkline" in captured.out.lower() or "▁▂▃▄▅▆▇█" in captured.out or "█" in captured.out

    def test_csv_path_directly(self, training_csv):
        """Should also accept a direct path to training_log.csv."""
        csv_file = training_csv / "training_log.csv"
        result = print_training_status(csv_file, verbose=False)
        assert result["total_train_games"] == 100

    def test_tolerates_malformed_rows(self, tmp_path):
        """CSV with a corrupt row should not crash."""
        csv_path = tmp_path / "training_log.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "tag", "value"])
            writer.writerow([1, "train/score", 1000])
            fh.write(".9468586471898316\n")   # corrupt row (matches real bug)
            writer.writerow([2, "train/score", 1100])
        result = print_training_status(tmp_path, verbose=False)
        assert result["total_train_games"] == 2


# ---------------------------------------------------------------------------
# CLI integration (parse_args wires the flags)
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_inspect_checkpoint_in_parse_args(self):
        from main import parse_args
        args = parse_args(["--inspect-checkpoint", "/some/path.npz"])
        assert args.inspect_checkpoint == "/some/path.npz"

    def test_training_status_in_parse_args(self):
        from main import parse_args
        args = parse_args(["--training-status", "/some/dir"])
        assert args.training_status == "/some/dir"

    def test_inspect_checkpoint_early_exit(self, dqn_checkpoint):
        """main() should return after inspection (not proceed to game loop)."""
        import main as m
        # Should NOT raise SystemExit or try to open a browser.
        m.main(["--inspect-checkpoint", str(dqn_checkpoint)])

    def test_training_status_early_exit(self, training_csv):
        """main() should return after printing status."""
        import main as m
        m.main(["--training-status", str(training_csv)])
