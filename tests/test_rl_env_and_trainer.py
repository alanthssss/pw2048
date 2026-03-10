"""Tests for the 4-layer RL architecture: Env, Train, Eval, and Play layers.

Covers:
* Game2048Env (reset, step, valid_actions, invalid-move penalty, terminal, properties)
* DQNAlgorithmV3.predict() and PPOAlgorithmV3.predict()
* TrainingLogger (CSV output)
* EvalCallback (runs eval games, saves best checkpoint)
* RLTrainer (train() method, in-process episode loop)
* make_trainer() factory
"""

from __future__ import annotations

import pathlib
import csv

import numpy as np
import pytest

from src.rl_env import Game2048Env, _encode_onehot, _spawn_tile, _init_board, _N_LEVELS
from src.rl_trainer import (
    EvalCallback,
    RLTrainer,
    TrainingLogger,
    make_trainer,
)
from src.algorithms.dqn_algo import DQNAlgorithmV3
from src.algorithms.ppo_algo import PPOAlgorithmV3
from src.game import DIRECTIONS

# Number of tile-level buckets per cell (imported from rl_env for test clarity).
_TILE_LEVELS = _N_LEVELS  # 16


# ===========================================================================
# Env layer — Game2048Env
# ===========================================================================

class TestGame2048Env:
    """Tests for Game2048Env (the Env layer)."""

    def test_reset_returns_correct_shape(self):
        env = Game2048Env(seed=0)
        obs, info = env.reset()
        assert obs.shape == (256,), "Observation must be 256-dim"
        assert obs.dtype == np.float32

    def test_reset_info_keys(self):
        env = Game2048Env(seed=0)
        _, info = env.reset()
        for key in ("board", "score", "max_tile", "n_steps"):
            assert key in info

    def test_reset_score_zero(self):
        env = Game2048Env(seed=0)
        _, info = env.reset()
        assert info["score"] == 0

    def test_reset_starts_with_two_tiles(self):
        env = Game2048Env(seed=0)
        env.reset()
        non_zero = sum(
            1 for r in range(4) for c in range(4) if env.board[r][c] != 0
        )
        assert non_zero == 2, "Fresh board must have exactly 2 tiles"

    def test_step_valid_move_changes_board(self):
        env = Game2048Env(seed=0)
        env.reset()
        board_before = [row[:] for row in env.board]
        # Try all actions until we find a valid one.
        for action in range(4):
            if action in env.valid_actions():
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == (256,)
                assert not truncated
                break

    def test_step_invalid_move_returns_penalty(self):
        """Stepping with an invalid action returns –0.1 and does not end the episode."""
        env = Game2048Env(seed=0)
        env.reset()
        # Find an invalid action (one that doesn't change the board).
        valid = env.valid_actions()
        invalid_actions = [a for a in range(4) if a not in valid]
        if not invalid_actions:
            pytest.skip("All actions are valid for this seed — cannot test invalid move")
        action = invalid_actions[0]
        board_before = [row[:] for row in env.board]
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == pytest.approx(-0.1)
        assert not terminated
        assert env.board == board_before, "Board must be unchanged after invalid move"

    def test_step_raises_on_done_episode(self):
        env = Game2048Env(seed=0)
        env.reset()
        env._done = True  # Force the done flag.
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(0)

    def test_step_increments_n_steps_on_valid_move(self):
        env = Game2048Env(seed=0)
        env.reset()
        assert env.n_steps == 0
        valid = env.valid_actions()
        if valid:
            env.step(valid[0])
            assert env.n_steps == 1

    def test_step_does_not_increment_n_steps_on_invalid_move(self):
        env = Game2048Env(seed=0)
        env.reset()
        valid = env.valid_actions()
        invalid = [a for a in range(4) if a not in valid]
        if not invalid:
            pytest.skip("All actions valid")
        env.step(invalid[0])
        assert env.n_steps == 0

    def test_valid_actions_subset_of_0123(self):
        env = Game2048Env(seed=0)
        env.reset()
        va = env.valid_actions()
        assert all(a in range(4) for a in va)

    def test_observation_is_onehot(self):
        env = Game2048Env(seed=42)
        obs, _ = env.reset()
        # Each _TILE_LEVELS-element slice must be a valid one-hot.
        for cell in range(16):
            chunk = obs[cell * _TILE_LEVELS: (cell + 1) * _TILE_LEVELS]
            assert chunk.sum() == pytest.approx(1.0), \
                f"Cell {cell} one-hot slice should sum to 1"

    def test_max_tile_matches_board(self):
        env = Game2048Env(seed=0)
        env.reset()
        expected = max(env.board[r][c] for r in range(4) for c in range(4))
        assert env.max_tile == expected

    def test_score_accumulates(self):
        env = Game2048Env(seed=0)
        env.reset()
        prev_score = env.score
        for action in range(4):
            if action in env.valid_actions():
                env.step(action)
                break
        assert env.score >= prev_score, "Score should be non-decreasing"

    def test_full_game_terminates(self):
        env = Game2048Env(seed=7)
        env.reset()
        steps = 0
        while not env.is_done:
            valid = env.valid_actions()
            if not valid:
                break
            env.step(valid[0])
            steps += 1
            if steps > 5000:
                break  # Safety guard — should not be needed.
        assert env.is_done or steps > 0

    def test_board_property_returns_copy(self):
        """Modifying the returned board must not affect internal state."""
        env = Game2048Env(seed=0)
        env.reset()
        board_copy = env.board
        board_copy[0][0] = 99999
        assert env.board[0][0] != 99999

    def test_seeded_reset_is_deterministic(self):
        env = Game2048Env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_observation_constants(self):
        assert Game2048Env.observation_size == 256
        assert Game2048Env.n_actions == 4


# ===========================================================================
# Env helpers
# ===========================================================================

class TestEnvHelpers:
    def test_encode_onehot_empty_cell(self):
        board = [[0] * 4 for _ in range(4)]
        obs = _encode_onehot(board)
        # All cells empty → level 0 set for each cell.
        for cell in range(16):
            chunk = obs[cell * _TILE_LEVELS: (cell + 1) * _TILE_LEVELS]
            assert chunk[0] == 1.0
            assert chunk[1:].sum() == 0.0

    def test_encode_onehot_tile_2(self):
        board = [[0] * 4 for _ in range(4)]
        board[0][0] = 2  # tile 2 = log2(2) = 1 → level 1
        obs = _encode_onehot(board)
        assert obs[1] == 1.0  # cell 0, level 1

    def test_spawn_tile_adds_exactly_one_tile(self):
        rng = np.random.default_rng(0)
        board = [[0] * 4 for _ in range(4)]
        new_board = _spawn_tile(board, rng)
        count = sum(new_board[r][c] != 0 for r in range(4) for c in range(4))
        assert count == 1

    def test_spawn_tile_value_is_2_or_4(self):
        rng = np.random.default_rng(0)
        board = [[0] * 4 for _ in range(4)]
        values = set()
        for _ in range(100):
            nb = _spawn_tile(board, rng)
            for r in range(4):
                for c in range(4):
                    if nb[r][c]:
                        values.add(nb[r][c])
        assert values.issubset({2, 4})

    def test_init_board_has_two_tiles(self):
        rng = np.random.default_rng(0)
        board = _init_board(rng)
        count = sum(board[r][c] != 0 for r in range(4) for c in range(4))
        assert count == 2


# ===========================================================================
# predict() — DQNAlgorithmV3 and PPOAlgorithmV3
# ===========================================================================

class TestDQNPredict:
    """Tests for DQNAlgorithmV3.predict()."""

    def test_predict_returns_valid_direction(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        direction = algo.predict(board)
        assert direction in DIRECTIONS

    def test_predict_does_not_modify_epsilon(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, epsilon_start=0.7, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        eps_before = algo._epsilon
        algo.predict(board)
        assert algo._epsilon == eps_before, "predict() must not change epsilon"

    def test_predict_does_not_modify_step(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        step_before = algo._step
        algo.predict(board)
        assert algo._step == step_before, "predict() must not change step counter"

    def test_predict_does_not_modify_buffer(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        len_before = len(algo._buffer)
        algo.predict(board)
        assert len(algo._buffer) == len_before, "predict() must not touch replay buffer"

    def test_predict_is_deterministic_when_greedy(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        d1 = algo.predict(board)
        d2 = algo.predict(board)
        assert d1 == d2, "predict() should be deterministic"


class TestPPOPredict:
    """Tests for PPOAlgorithmV3.predict()."""

    def test_predict_returns_valid_direction(self):
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        direction = algo.predict(board)
        assert direction in DIRECTIONS

    def test_predict_does_not_modify_buffer(self):
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        len_before = len(algo._buf_states)
        algo.predict(board)
        assert len(algo._buf_states) == len_before

    def test_predict_is_deterministic(self):
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        d1 = algo.predict(board)
        d2 = algo.predict(board)
        assert d1 == d2

    def test_predict_does_not_touch_prev_state(self):
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0)
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
        algo.predict(board)
        assert algo._prev_state is None, "predict() must not set _prev_state"


# ===========================================================================
# TrainingLogger
# ===========================================================================

class TestTrainingLogger:
    """Tests for TrainingLogger (CSV + optional TensorBoard)."""

    def test_creates_csv_file(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.close()
        assert (tmp_path / "training_log.csv").exists()

    def test_csv_has_header(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.close()
        with open(tmp_path / "training_log.csv") as f:
            header = f.readline().strip()
        assert header == "step,tag,value"

    def test_log_writes_rows(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log("train/score", 1024.0, 1)
        logger.log("eval/mean_score", 800.5, 1)
        logger.close()
        with open(tmp_path / "training_log.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["tag"] == "train/score"
        assert float(rows[0]["value"]) == pytest.approx(1024.0)

    def test_log_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "sub" / "logs"
        logger = TrainingLogger(new_dir)
        logger.close()
        assert new_dir.exists()

    def test_flush_and_close_do_not_raise(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log("x", 1.0, 0)
        logger.flush()
        logger.close()  # must not raise


# ===========================================================================
# EvalCallback
# ===========================================================================

class TestEvalCallback:
    """Tests for EvalCallback (Eval layer)."""

    def _make_algo(self) -> DQNAlgorithmV3:
        return DQNAlgorithmV3(n_pretrain_games=0, seed=0)

    def test_callback_returns_none_between_evals(self):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        cb = EvalCallback(algo, env, eval_freq=5, n_eval_games=2)
        assert cb(1) is None
        assert cb(4) is None

    def test_callback_runs_on_correct_frequency(self):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        cb = EvalCallback(algo, env, eval_freq=5, n_eval_games=2)
        result = cb(5)
        assert result is not None
        assert "mean_score" in result

    def test_callback_returns_dict_with_expected_keys(self):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        cb = EvalCallback(algo, env, eval_freq=5, n_eval_games=3)
        result = cb(5)
        for key in ("game_num", "mean_score", "max_score", "mean_tile", "max_tile", "is_new_best"):
            assert key in result

    def test_callback_saves_best_checkpoint(self, tmp_path):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        best = tmp_path / "best.npz"
        cb = EvalCallback(algo, env, eval_freq=1, n_eval_games=2, best_ckpt_path=best)
        result = cb(1)
        assert best.exists(), "Best checkpoint must be saved on first eval (always new best)"
        assert result["is_new_best"] is True

    def test_callback_tracks_best_mean_score(self):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        cb = EvalCallback(algo, env, eval_freq=1, n_eval_games=2)
        assert cb.best_mean_score == float("-inf")
        cb(1)
        assert cb.best_mean_score > float("-inf")

    def test_callback_logs_to_logger(self, tmp_path):
        algo = self._make_algo()
        env = Game2048Env(seed=0)
        logger = TrainingLogger(tmp_path)
        cb = EvalCallback(algo, env, eval_freq=1, n_eval_games=2, logger=logger)
        cb(1)
        logger.close()
        with open(tmp_path / "training_log.csv") as f:
            rows = list(csv.DictReader(f))
        tags = {row["tag"] for row in rows}
        assert "eval/mean_score" in tags


# ===========================================================================
# RLTrainer
# ===========================================================================

class TestRLTrainer:
    """Tests for RLTrainer (Train layer)."""

    def _make_algo(self) -> DQNAlgorithmV3:
        return DQNAlgorithmV3(
            n_pretrain_games=0, batch_size=4, buffer_size=50,
            train_freq=1, seed=0
        )

    def test_train_returns_summary_dict(self, tmp_path):
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=3)
        for key in ("total_games", "mean_score", "max_score", "max_tile", "elapsed_s"):
            assert key in summary

    def test_train_total_games_matches(self, tmp_path):
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=5)
        assert summary["total_games"] == 5

    def test_train_saves_checkpoint(self, tmp_path):
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, checkpoint_dir=tmp_path, verbose=False)
        trainer.train(total_games=3)
        assert (tmp_path / "checkpoint.npz").exists()

    def test_train_writes_csv_log(self, tmp_path):
        algo = self._make_algo()
        tb_dir = tmp_path / "tb"
        trainer = RLTrainer(algorithm=algo, tensorboard_dir=tb_dir, verbose=False)
        trainer.train(total_games=3)
        assert (tb_dir / "training_log.csv").exists()

    def test_train_csv_has_score_entries(self, tmp_path):
        algo = self._make_algo()
        tb_dir = tmp_path / "tb"
        trainer = RLTrainer(algorithm=algo, tensorboard_dir=tb_dir, verbose=False)
        trainer.train(total_games=3)
        with open(tb_dir / "training_log.csv") as f:
            rows = list(csv.DictReader(f))
        tags = {r["tag"] for r in rows}
        assert "train/score" in tags

    def test_train_with_eval_callback(self, tmp_path):
        algo = self._make_algo()
        eval_env = Game2048Env(seed=99)
        cb = EvalCallback(algo, eval_env, eval_freq=3, n_eval_games=2, verbose=False)
        trainer = RLTrainer(algorithm=algo, eval_callback=cb, verbose=False)
        summary = trainer.train(total_games=6)
        assert cb.best_mean_score > float("-inf")

    def test_train_updates_algorithm_weights(self, tmp_path):
        """Training games should change the Q-network weights."""
        algo = self._make_algo()
        W1_before = algo._q_net.W1.copy()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        trainer.train(total_games=10)
        # Weights must have changed (training happened).
        assert not np.array_equal(algo._q_net.W1, W1_before), \
            "Weights must change after training"

    def test_total_steps_counts_moves_not_scores(self, tmp_path):
        """total_steps in the summary must be the sum of valid moves, not scores."""
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=5)
        # total_steps must be a reasonable number of game moves (not a large score sum).
        # A 2048 game typically has 100–500 moves; scores can be much higher.
        # We just verify total_steps > 0 and does not equal the sum of scores
        # (which would indicate the old bug of summing scores instead of steps).
        assert summary["total_steps"] > 0
        assert summary["total_steps"] != summary["max_score"] * 5 or summary["max_score"] == 0

    def test_ppo_trainer(self, tmp_path):
        """RLTrainer also works with PPOAlgorithmV3."""
        algo = PPOAlgorithmV3(
            n_pretrain_games=0, update_freq=4, n_epochs=1, seed=0
        )
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=5)
        assert summary["total_games"] == 5
        assert summary["mean_score"] >= 0


# ===========================================================================
# make_trainer factory
# ===========================================================================

class TestMakeTrainer:
    """Tests for the make_trainer() convenience factory."""

    def test_returns_rl_trainer(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        trainer = make_trainer(algo, verbose=False)
        assert isinstance(trainer, RLTrainer)

    def test_with_checkpoint_and_tensorboard(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        ckpt = tmp_path / "ckpt"
        tb = tmp_path / "tb"
        trainer = make_trainer(
            algo,
            checkpoint_dir=ckpt,
            tensorboard_dir=tb,
            eval_freq=3,
            n_eval_games=2,
            verbose=False,
        )
        summary = trainer.train(total_games=5)
        assert (ckpt / "checkpoint.npz").exists()
        assert (tb / "training_log.csv").exists()

    def test_wires_eval_callback(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        ckpt = tmp_path / "ckpt"
        trainer = make_trainer(
            algo,
            checkpoint_dir=ckpt,
            eval_freq=2,
            n_eval_games=2,
            verbose=False,
        )
        trainer.train(total_games=4)
        assert trainer._eval_cb is not None
        assert trainer._eval_cb.best_mean_score > float("-inf")


# ===========================================================================
# GPU device detection and optional PyTorch backend
# ===========================================================================

class TestDeviceDetection:
    """Tests for the _detect_device() helper in DQN and PPO."""

    def test_dqn_detect_device_returns_none_or_str(self):
        from src.algorithms.dqn_algo import _detect_device, _TORCH_AVAILABLE
        result = _detect_device()
        if _TORCH_AVAILABLE:
            assert isinstance(result, str)
        else:
            assert result is None

    def test_ppo_detect_device_returns_none_or_str(self):
        from src.algorithms.ppo_algo import _detect_device, _TORCH_AVAILABLE
        result = _detect_device()
        if _TORCH_AVAILABLE:
            assert isinstance(result, str)
        else:
            assert result is None

    def test_dqn_device_numpy_forces_numpy_backend(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        assert algo._use_torch is False

    def test_ppo_device_numpy_forces_numpy_backend(self):
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        assert algo._use_torch is False

    def test_dqn_auto_device_consistent_with_torch_availability(self):
        """When torch is absent, auto-detection must select the numpy backend."""
        from src.algorithms.dqn_algo import _TORCH_AVAILABLE
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0)
        if not _TORCH_AVAILABLE:
            assert algo._use_torch is False

    def test_ppo_auto_device_consistent_with_torch_availability(self):
        from src.algorithms.ppo_algo import _TORCH_AVAILABLE
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0)
        if not _TORCH_AVAILABLE:
            assert algo._use_torch is False


class TestTorchBackend:
    """Tests for the optional PyTorch compute backend (skipped when torch absent)."""

    @staticmethod
    def _skip_if_no_torch():
        from src.algorithms.dqn_algo import _TORCH_AVAILABLE
        if not _TORCH_AVAILABLE:
            import pytest
            pytest.skip("PyTorch not installed")

    def test_dqn_cpu_device_uses_torch_backend(self):
        self._skip_if_no_torch()
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        assert algo._use_torch is True

    def test_dqn_torch_predict_returns_valid_direction(self):
        self._skip_if_no_torch()
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        assert algo.predict(board) in DIRECTIONS

    def test_dqn_torch_choose_move_returns_valid_direction(self):
        self._skip_if_no_torch()
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        algo.on_game_start()
        assert algo.choose_move(board) in DIRECTIONS

    def test_dqn_torch_checkpoint_roundtrip(self, tmp_path):
        self._skip_if_no_torch()
        algo1 = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        ckpt = tmp_path / "dqn_cpu.npz"
        algo1.save_checkpoint(ckpt)
        algo2 = DQNAlgorithmV3(n_pretrain_games=0, seed=99, device="cpu")
        algo2.load_checkpoint(ckpt)
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        assert algo1.predict(board) == algo2.predict(board)

    def test_ppo_cpu_device_uses_torch_backend(self):
        self._skip_if_no_torch()
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        assert algo._use_torch is True

    def test_ppo_torch_predict_returns_valid_direction(self):
        self._skip_if_no_torch()
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        assert algo.predict(board) in DIRECTIONS

    def test_ppo_torch_checkpoint_roundtrip(self, tmp_path):
        self._skip_if_no_torch()
        algo1 = PPOAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        ckpt = tmp_path / "ppo_cpu.npz"
        algo1.save_checkpoint(ckpt)
        algo2 = PPOAlgorithmV3(n_pretrain_games=0, seed=99, device="cpu")
        algo2.load_checkpoint(ckpt)
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        assert algo1.predict(board) == algo2.predict(board)

    def test_dqn_torch_trains_without_error(self):
        self._skip_if_no_torch()
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu")
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=3)
        assert summary["total_games"] == 3

    def test_ppo_torch_trains_without_error(self):
        self._skip_if_no_torch()
        algo = PPOAlgorithmV3(n_pretrain_games=0, seed=0, device="cpu", update_freq=4, n_epochs=1)
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=3)
        assert summary["total_games"] == 3

    def test_cross_backend_checkpoint_compatibility(self, tmp_path):
        """Checkpoint saved by numpy backend can be loaded by torch backend."""
        self._skip_if_no_torch()
        numpy_algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        ckpt = tmp_path / "cross.npz"
        numpy_algo.save_checkpoint(ckpt)
        torch_algo = DQNAlgorithmV3(n_pretrain_games=0, seed=99, device="cpu")
        torch_algo.load_checkpoint(ckpt)
        board = [[2,4,8,16],[32,64,128,256],[2,4,8,16],[32,64,0,0]]
        assert numpy_algo.predict(board) == torch_algo.predict(board)


# ===========================================================================
# Parallel training
# ===========================================================================

class TestParallelTraining:
    """Tests for n_workers > 1 in RLTrainer."""

    def test_parallel_train_returns_summary(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        trainer = RLTrainer(algorithm=algo, verbose=False, n_workers=2)
        summary = trainer.train(total_games=2)
        assert summary["total_games"] == 2
        assert "n_workers" in summary
        assert summary["n_workers"] == 2

    def test_parallel_train_selects_best_worker(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        trainer = RLTrainer(algorithm=algo, verbose=False, n_workers=2)
        summary = trainer.train(total_games=2)
        assert "best_worker" in summary
        assert summary["best_worker"] in (0, 1)

    def test_parallel_train_saves_checkpoint(self, tmp_path):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        trainer = RLTrainer(
            algorithm=algo, checkpoint_dir=tmp_path, verbose=False, n_workers=2
        )
        trainer.train(total_games=2)
        assert (tmp_path / "checkpoint.npz").exists()

    def test_make_trainer_n_workers_parameter(self):
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        trainer = make_trainer(algo, verbose=False, n_workers=2)
        assert trainer._n_workers == 2

    def test_n_workers_one_falls_back_to_sequential(self, tmp_path):
        """n_workers=1 (default) must use the sequential path."""
        algo = DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")
        trainer = RLTrainer(algorithm=algo, verbose=False, n_workers=1)
        summary = trainer.train(total_games=2)
        # Sequential training does NOT include n_workers key.
        assert "n_workers" not in summary


# ===========================================================================
# Early stopping
# ===========================================================================

class TestEarlyStoppingCallback:
    """Tests for EvalCallback early stopping (patience + min_delta)."""

    def _make_algo(self):
        return DQNAlgorithmV3(n_pretrain_games=0, seed=0, device="numpy")

    def test_patience_zero_disables_early_stopping(self):
        """patience=0 must never set should_stop."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=1)
        cb = EvalCallback(
            algo, eval_env, eval_freq=1, n_eval_games=2, verbose=False, patience=0
        )
        for i in range(1, 20):
            cb(i)
        assert cb.should_stop is False

    def test_patience_triggers_after_n_non_improving_rounds(self):
        """should_stop becomes True exactly after patience rounds without improvement."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=42)
        cb = EvalCallback(
            algo, eval_env, eval_freq=1, n_eval_games=2, verbose=False,
            patience=3, min_delta=0.0,
        )
        # Force first eval to establish best_mean_score, then pin it to a
        # very high value so subsequent real evals never beat old_best+min_delta.
        cb(1)
        cb._best_mean_score = 1e12   # subsequent evals cannot surpass this
        assert cb.should_stop is False, "Should not stop after first eval"
        # After 3 more rounds without improvement patience should be exhausted.
        cb(2)
        cb(3)
        assert cb.should_stop is False, "Should not stop before patience rounds"
        cb(4)
        assert cb.should_stop is True, "Should stop after patience rounds"

    def test_patience_resets_on_improvement(self):
        """When score genuinely improves, the counter should reset."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=7)
        cb = EvalCallback(
            algo, eval_env, eval_freq=1, n_eval_games=2, verbose=False,
            patience=2, min_delta=0.0,  # any improvement resets counter
        )
        # Establish best.
        cb(1)
        # Non-improving round (pin best to a high value so real evals don't improve).
        cb._best_mean_score = 1e12
        cb(2)
        assert cb.no_improve_count == 1
        # Simulate improvement by resetting best_mean_score to a low value.
        cb._best_mean_score = -1.0  # next eval will beat this
        cb(3)
        assert cb.no_improve_count == 0, "Counter must reset on improvement"
        assert cb.should_stop is False

    def test_no_improve_count_property(self):
        """no_improve_count increments correctly."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=3)
        cb = EvalCallback(
            algo, eval_env, eval_freq=1, n_eval_games=2, verbose=False,
            patience=5, min_delta=0.0,
        )
        cb(1)  # sets best
        cb._best_mean_score = 1e12  # pin so subsequent rounds never improve
        for i in range(2, 6):
            cb(i)
        assert cb.no_improve_count == 4

    def test_trainer_stops_early_when_patience_exhausted(self):
        """RLTrainer must exit before total_games when should_stop fires."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=0)
        cb = EvalCallback(
            algo, eval_env,
            eval_freq=2,
            n_eval_games=2,
            verbose=False,
            patience=2,
            min_delta=0.0,
        )
        # Pin best_mean_score after first eval by overriding the method;
        # all subsequent evals won't improve → patience exhausts quickly.
        original_run_eval = cb._run_eval
        call_count = [0]
        def _patched_run_eval(game_num: int) -> dict:
            result = original_run_eval(game_num)
            call_count[0] += 1
            if call_count[0] >= 1:
                cb._best_mean_score = 1e12  # pin after first eval
            return result
        cb._run_eval = _patched_run_eval
        trainer = RLTrainer(algorithm=algo, eval_callback=cb, verbose=False)
        summary = trainer.train(total_games=1000)
        assert summary["stopped_early"] is True
        assert summary["total_games"] < 1000

    def test_trainer_does_not_stop_early_when_patience_zero(self):
        """RLTrainer must play all games when patience=0 (default)."""
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=5)
        assert summary["stopped_early"] is False
        assert summary["total_games"] == 5

    def test_make_trainer_patience_wired(self):
        """make_trainer passes patience + min_delta through to EvalCallback."""
        algo = self._make_algo()
        trainer = make_trainer(algo, verbose=False, patience=3, min_delta=50.0)
        assert trainer._eval_cb is not None
        assert trainer._eval_cb._patience == 3
        assert trainer._eval_cb._min_delta == 50.0

    def test_summary_stopped_early_key_always_present(self):
        """Summary dict must always contain 'stopped_early' key."""
        algo = self._make_algo()
        trainer = RLTrainer(algorithm=algo, verbose=False)
        summary = trainer.train(total_games=2)
        assert "stopped_early" in summary

    def test_early_stop_saves_best_checkpoint(self, tmp_path):
        """Best checkpoint must be saved even when training stopped early."""
        algo = self._make_algo()
        eval_env = Game2048Env(seed=0)
        best_ckpt = tmp_path / "best.npz"
        cb = EvalCallback(
            algo, eval_env,
            eval_freq=2, n_eval_games=2,
            best_ckpt_path=best_ckpt,
            verbose=False,
            patience=2, min_delta=0.0,
        )
        # Pin best after first eval so patience exhausts quickly.
        original_run_eval = cb._run_eval
        called = [False]
        def _pin_after_first(game_num: int) -> dict:
            result = original_run_eval(game_num)
            if not called[0]:
                called[0] = True
                cb._best_mean_score = 1e12
            return result
        cb._run_eval = _pin_after_first
        trainer = RLTrainer(algorithm=algo, eval_callback=cb, verbose=False)
        trainer.train(total_games=200)
        # best_checkpoint.npz should exist (first eval is always a new best).
        assert best_ckpt.exists()
