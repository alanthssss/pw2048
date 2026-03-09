"""Train and Eval layers: structured RL training loop for 2048.

Architecture overview
---------------------
The four layers of the RL training stack are:

* **Env** — :class:`src.rl_env.Game2048Env` (pure-Python, no browser).
* **Train** — :class:`RLTrainer` (this module): runs episodes via the Env and
  delegates learning to the algorithm's existing ``choose_move()`` method.
* **Eval** — :class:`EvalCallback` (this module): periodic evaluation over
  deterministic (greedy) episodes; saves the best checkpoint.
* **Play** — :mod:`src.runner` / Playwright browser mode: benchmark and
  demonstration using the trained policy.

Logging
-------
:class:`TrainingLogger` writes scalar metrics to a CSV file so you can
plot training curves with any tool.  When the ``tensorboard`` package is
installed (``pip install tensorboard``), it also writes ``.tfevents`` files
that can be visualised live with::

    tensorboard --logdir <tensorboard-dir>

If ``tensorboard`` is not installed the logger silently falls back to CSV
only — no crash, no warning.

Usage example
-------------
.. code-block:: python

    from src.rl_env import Game2048Env
    from src.rl_trainer import RLTrainer
    from src.algorithms.dqn_algo import DQNAlgorithmV3

    algo = DQNAlgorithmV3(n_pretrain_games=50)
    trainer = RLTrainer(
        algorithm=algo,
        checkpoint_dir="checkpoints",
        tensorboard_dir="tb_logs",
        eval_freq=50,
        n_eval_games=20,
    )
    summary = trainer.train(total_games=2000)
    print(summary)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np

from src.rl_env import Game2048Env
from src.game import DIRECTIONS

if TYPE_CHECKING:
    from src.algorithms.base import BaseAlgorithm


# ---------------------------------------------------------------------------
# TrainingLogger — CSV + optional TensorBoard
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Write scalar training metrics to CSV and optionally TensorBoard.

    The CSV file is always created at ``log_dir / "training_log.csv"`` and
    contains three columns: ``step``, ``tag``, ``value``.

    If the ``tensorboard`` package is installed (``pip install tensorboard``),
    a ``.tfevents`` file is also written to the same directory, making the log
    compatible with ``tensorboard --logdir <log_dir>``.

    Parameters
    ----------
    log_dir:
        Directory where all log files are written.  Created if absent.
    """

    def __init__(self, log_dir: Union[str, Path]) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # CSV writer (always available).
        csv_path = self._dir / "training_log.csv"
        self._csv_fh = csv_path.open("w", newline="")
        self._csv = csv.writer(self._csv_fh)
        self._csv.writerow(["step", "tag", "value"])

        # Optional TensorBoard writer (lazy import, graceful fallback).
        # Priority: PyTorch SummaryWriter → standalone tensorboard SummaryWriter
        # → CSV only.
        self._tb_writer: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

            self._tb_writer = SummaryWriter(log_dir=str(self._dir))
        except ImportError:
            try:
                # Standalone `tensorboard` package (pip install tensorboard).
                # SummaryWriter in the `tensorboard.summary.writer` module
                # writes .tfevents files without requiring TensorFlow.
                from tensorboard.summary.writer.event_file_writer import (  # type: ignore[import]
                    EventFileWriter,
                )

                self._tb_writer = _TBWriter(self._dir)
            except ImportError:
                pass  # TensorBoard not installed — CSV only.


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, tag: str, value: float, step: int) -> None:
        """Record a scalar metric.

        Parameters
        ----------
        tag:
            Metric name, e.g. ``"train/score"`` or ``"eval/mean_score"``.
        value:
            Scalar value to record.
        step:
            Global training step (e.g. game number).
        """
        self._csv.writerow([step, tag, value])
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(tag, value, global_step=step)
            except Exception:  # noqa: BLE001
                pass

    def flush(self) -> None:
        """Flush all buffered writes to disk."""
        self._csv_fh.flush()
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """Flush and close all file handles."""
        self.flush()
        self._csv_fh.close()
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:  # noqa: BLE001
                pass


class _TBWriter:
    """Thin wrapper that writes TensorBoard scalar summaries using the
    standalone ``tensorboard`` package (no PyTorch or TensorFlow required).

    It uses :class:`tensorboard.summary.writer.event_file_writer.EventFileWriter`
    and builds minimal ``tf.Event`` / summary protos by hand via the
    ``tensorboard.compat.proto`` package that ships inside ``tensorboard``.

    If proto construction fails for any reason the write is silently dropped
    (the CSV log is always the primary record).
    """

    def __init__(self, log_dir: Path) -> None:
        from tensorboard.summary.writer.event_file_writer import (  # type: ignore[import]
            EventFileWriter,
        )
        self._writer = EventFileWriter(str(log_dir))
        self._step_map: dict[str, int] = {}

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        try:
            from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore[import]
            from tensorboard.compat.proto.event_pb2 import Event       # type: ignore[import]
            import time as _time

            summary = Summary()
            summary.value.add(tag=tag, simple_value=float(value))
            event = Event(
                wall_time=_time.time(),
                step=global_step,
                summary=summary,
            )
            self._writer.write_event(event)
        except Exception:  # noqa: BLE001
            pass  # Silently skip — CSV is always written.

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# EvalCallback — Eval layer
# ---------------------------------------------------------------------------

class EvalCallback:
    """Periodically evaluate the algorithm and save the best checkpoint.

    The callback runs *n_eval_games* episodes using the algorithm's
    :meth:`predict` method (greedy, no exploration) and logs the average
    and maximum scores.  When a new best average score is achieved the
    weights are saved to *best_ckpt_path*.

    This is the **Eval layer** of the Env / Train / Eval / Play stack.

    Parameters
    ----------
    algorithm:
        The algorithm to evaluate.  Must implement ``predict(board) → str``.
    eval_env:
        A :class:`~src.rl_env.Game2048Env` instance used for evaluation.
    eval_freq:
        Evaluate every *eval_freq* training games.
    n_eval_games:
        Number of games per evaluation round.
    best_ckpt_path:
        Where to save the weights when a new best score is reached.
        If ``None``, checkpoints are not saved.
    logger:
        :class:`TrainingLogger` for recording eval metrics.  If ``None``,
        metrics are printed to stdout only.
    verbose:
        If ``True`` (default), print a summary after each evaluation.
    """

    def __init__(
        self,
        algorithm: "BaseAlgorithm",
        eval_env: Game2048Env,
        eval_freq: int = 50,
        n_eval_games: int = 20,
        best_ckpt_path: Optional[Union[str, Path]] = None,
        logger: Optional[TrainingLogger] = None,
        verbose: bool = True,
    ) -> None:
        self._algo = algorithm
        self._env = eval_env
        self._eval_freq = eval_freq
        self._n_eval = n_eval_games
        self._best_path = Path(best_ckpt_path) if best_ckpt_path else None
        self._logger = logger
        self._verbose = verbose
        self._best_mean_score: float = float("-inf")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def eval_freq(self) -> int:
        """Evaluation frequency in training games."""
        return self._eval_freq

    @property
    def best_mean_score(self) -> float:
        """Best mean evaluation score seen so far."""
        return self._best_mean_score

    def __call__(self, game_num: int) -> Optional[dict]:
        """Run evaluation if *game_num* is a multiple of *eval_freq*.

        Parameters
        ----------
        game_num:
            Current training game number (1-indexed).

        Returns
        -------
        dict or None
            Evaluation result dict when evaluation ran; ``None`` otherwise.
        """
        if game_num % self._eval_freq != 0:
            return None
        return self._run_eval(game_num)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_eval(self, game_num: int) -> dict:
        scores: List[int] = []
        max_tiles: List[int] = []
        has_predict = hasattr(self._algo, "predict")

        for _ in range(self._n_eval):
            self._env.reset()
            while not self._env.is_done:
                board = self._env.board
                if has_predict:
                    direction = self._algo.predict(board)
                else:
                    direction = self._algo.choose_move(board)
                action = DIRECTIONS.index(direction)
                self._env.step(action)
            scores.append(self._env.score)
            max_tiles.append(self._env.max_tile)

        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        mean_tile = float(np.mean(max_tiles))
        max_tile = int(np.max(max_tiles))
        is_new_best = mean_score > self._best_mean_score

        if is_new_best:
            self._best_mean_score = mean_score
            if self._best_path is not None and hasattr(self._algo, "save_checkpoint"):
                self._best_path.parent.mkdir(parents=True, exist_ok=True)
                self._algo.save_checkpoint(self._best_path)

        if self._logger is not None:
            self._logger.log("eval/mean_score", mean_score, game_num)
            self._logger.log("eval/max_score",  max_score,  game_num)
            self._logger.log("eval/mean_tile",  mean_tile,  game_num)
            self._logger.log("eval/max_tile",   float(max_tile), game_num)

        if self._verbose:
            star = " ★ new best" if is_new_best else ""
            print(
                f"  [eval @ game {game_num:>5}]  "
                f"mean_score={mean_score:>7.0f}  "
                f"max_score={max_score:>7.0f}  "
                f"max_tile={max_tile:>4}{star}"
            )

        return {
            "game_num": game_num,
            "mean_score": mean_score,
            "max_score": max_score,
            "mean_tile": mean_tile,
            "max_tile": max_tile,
            "is_new_best": is_new_best,
        }


# ---------------------------------------------------------------------------
# RLTrainer — Train layer
# ---------------------------------------------------------------------------

class RLTrainer:
    """Fast in-process RL training loop (no browser required).

    This is the **Train layer** of the Env / Train / Eval / Play stack.
    It drives training episodes using :class:`~src.rl_env.Game2048Env`
    rather than Playwright so each game runs 10–50× faster.

    During training the algorithm's existing :meth:`choose_move` is called
    for every board state.  This means ``DQNAlgorithmV3`` and
    ``PPOAlgorithmV3`` perform their internal experience-collection and
    gradient updates transparently — the Trainer does not need to know about
    replay buffers or PPO epochs.

    After training, the algorithm can be benchmarked in the browser using
    the existing ``python main.py --algorithm dqn`` workflow.

    Parameters
    ----------
    algorithm:
        A learning algorithm (e.g. :class:`~src.algorithms.dqn_algo.DQNAlgorithmV3`).
    checkpoint_dir:
        Directory where the latest weights are saved after every game.
        Set to ``None`` to disable checkpointing.
    tensorboard_dir:
        Directory for TensorBoard and CSV training logs.  Set to ``None``
        to disable logging entirely.
    eval_callback:
        An :class:`EvalCallback` instance.  Set to ``None`` to skip
        periodic evaluation.
    verbose:
        If ``True`` (default), print per-game statistics during training.
    """

    def __init__(
        self,
        algorithm: "BaseAlgorithm",
        *,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        tensorboard_dir: Optional[Union[str, Path]] = None,
        eval_callback: Optional[EvalCallback] = None,
        verbose: bool = True,
    ) -> None:
        self._algo = algorithm
        self._ckpt_dir: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
        self._logger: Optional[TrainingLogger] = (
            TrainingLogger(tensorboard_dir) if tensorboard_dir else None
        )
        self._eval_cb = eval_callback
        self._verbose = verbose
        self._env = Game2048Env()

        if self._ckpt_dir is not None:
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, total_games: int) -> dict:
        """Run *total_games* training episodes.

        Each episode uses the in-process :class:`~src.rl_env.Game2048Env`
        rather than a browser, making training substantially faster.

        Parameters
        ----------
        total_games:
            Number of training games to play.

        Returns
        -------
        dict
            Training summary with keys:
            ``total_games``, ``mean_score``, ``max_score``, ``max_tile``,
            ``total_steps``, ``elapsed_s``, ``best_eval_score``.
        """
        scores: List[int] = []
        max_tiles: List[int] = []
        t0 = time.perf_counter()
        n_steps_list: List[int] = []

        for game_num in range(1, total_games + 1):
            result = self._run_episode()
            scores.append(result["score"])
            max_tiles.append(result["max_tile"])
            n_steps_list.append(result["n_steps"])

            if self._verbose and (game_num % 50 == 0 or game_num == 1):
                print(
                    f"  [train] game {game_num:>5}/{total_games}  "
                    f"score={result['score']:>6}  "
                    f"max_tile={result['max_tile']:>4}  "
                    f"steps={result['n_steps']:>4}"
                )

            if self._logger is not None:
                self._logger.log("train/score",    float(result["score"]),    game_num)
                self._logger.log("train/max_tile",  float(result["max_tile"]), game_num)
                self._logger.log("train/n_steps",   float(result["n_steps"]),  game_num)
                epsilon = getattr(self._algo, "_epsilon", None)
                if epsilon is not None:
                    self._logger.log("train/epsilon", epsilon, game_num)

            # Periodic evaluation.
            if self._eval_cb is not None:
                self._eval_cb(game_num)
                if self._logger is not None:
                    self._logger.flush()

            # Latest checkpoint.
            if self._ckpt_dir is not None and hasattr(self._algo, "save_checkpoint"):
                ckpt = self._ckpt_dir / "checkpoint.npz"
                self._algo.save_checkpoint(ckpt)

        elapsed = time.perf_counter() - t0
        mean_score = float(np.mean(scores)) if scores else 0.0

        summary = {
            "total_games": total_games,
            "mean_score": mean_score,
            "max_score": int(np.max(scores)) if scores else 0,
            "max_tile": int(np.max(max_tiles)) if max_tiles else 0,
            "total_steps": sum(n_steps_list),
            "elapsed_s": round(elapsed, 2),
            "best_eval_score": (
                self._eval_cb.best_mean_score if self._eval_cb else float("nan")
            ),
        }

        if self._logger is not None:
            self._logger.log("summary/mean_score",    summary["mean_score"],    total_games)
            self._logger.log("summary/max_score",     float(summary["max_score"]),  total_games)
            self._logger.close()

        if self._verbose:
            print(
                f"\n  Training complete — {total_games} games in {elapsed:.1f}s  "
                f"mean_score={mean_score:.0f}  "
                f"max_score={summary['max_score']}"
            )
            if self._eval_cb is not None and self._eval_cb.best_mean_score > float("-inf"):
                print(
                    f"  Best eval mean_score = {self._eval_cb.best_mean_score:.0f}"
                )

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(self) -> dict:
        """Run one complete training episode and return statistics."""
        self._env.reset()
        self._algo.on_game_start()

        while not self._env.is_done:
            board = self._env.board
            direction = self._algo.choose_move(board)
            action = DIRECTIONS.index(direction)
            self._env.step(action)

        return {
            "score":    self._env.score,
            "max_tile": self._env.max_tile,
            "n_steps":  self._env.n_steps,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_trainer(
    algorithm: "BaseAlgorithm",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    tensorboard_dir: Optional[Union[str, Path]] = None,
    eval_freq: int = 50,
    n_eval_games: int = 20,
    verbose: bool = True,
) -> RLTrainer:
    """Build an :class:`RLTrainer` with a pre-wired :class:`EvalCallback`.

    Parameters
    ----------
    algorithm:
        The learning algorithm to train.
    checkpoint_dir:
        Directory for the latest *and* best checkpoints.  ``None`` → no
        checkpointing.
    tensorboard_dir:
        Directory for TensorBoard / CSV logs.  ``None`` → no logging.
    eval_freq:
        Evaluate every *eval_freq* training games.
    n_eval_games:
        Number of games per evaluation round.
    verbose:
        Print progress to stdout.

    Returns
    -------
    RLTrainer
        Ready-to-use trainer.  Call :meth:`RLTrainer.train` to start.
    """
    logger: Optional[TrainingLogger] = (
        TrainingLogger(tensorboard_dir) if tensorboard_dir else None
    )

    best_ckpt: Optional[Path] = (
        Path(checkpoint_dir) / "best_checkpoint.npz"
        if checkpoint_dir else None
    )

    eval_env = Game2048Env()
    eval_cb = EvalCallback(
        algorithm=algorithm,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_games=n_eval_games,
        best_ckpt_path=best_ckpt,
        logger=logger,
        verbose=verbose,
    )

    return RLTrainer(
        algorithm=algorithm,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        eval_callback=eval_cb,
        verbose=verbose,
    )
