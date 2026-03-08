"""Env layer: pure-Python gym-style 2048 environment for RL training.

This module provides :class:`Game2048Env`, which is the **Env layer** of the
Env / Train / Eval / Play stack.  It exposes a standard ``(reset, step,
valid_actions)`` interface so the Train and Eval layers never have to touch
the raw game logic.

The environment is entirely browser-free: it uses the same in-process move
simulation already present in :mod:`src.algorithms.greedy_algo`.  This makes
training 10–50× faster than the Playwright-based runner.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.algorithms.greedy_algo import _boards_equal, simulate_move
from src.game import DIRECTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of tile-level buckets per board cell.
#: Level 0 = empty; 1–15 = tiles 2¹–2¹⁵.
_N_LEVELS = 16

#: Flattened one-hot state dimension (16 cells × 16 levels = 256).
_OBS_SIZE = 16 * _N_LEVELS


# ---------------------------------------------------------------------------
# State encoding (one-hot)
# ---------------------------------------------------------------------------

def _encode_onehot(board: List[List[int]]) -> np.ndarray:
    """Encode *board* as a 256-dim float32 one-hot vector.

    Each of the 16 cells contributes a 16-element one-hot slice.  Level 0
    represents an empty cell; levels 1–15 represent tile values 2–32768.
    """
    out = np.zeros(_OBS_SIZE, dtype=np.float32)
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            cell_idx = r * 4 + c
            level = 0 if v == 0 else min(int(math.log2(v)), _N_LEVELS - 1)
            out[cell_idx * _N_LEVELS + level] = 1.0
    return out


# ---------------------------------------------------------------------------
# Tile spawning
# ---------------------------------------------------------------------------

def _spawn_tile(
    board: List[List[int]], rng: np.random.Generator
) -> List[List[int]]:
    """Return a copy of *board* with one new random tile (90 % → 2, 10 % → 4)."""
    empty = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empty:
        return [row[:] for row in board]
    new_board = [row[:] for row in board]
    r, c = empty[int(rng.integers(len(empty)))]
    new_board[r][c] = 2 if rng.random() < 0.9 else 4
    return new_board


def _init_board(rng: np.random.Generator) -> List[List[int]]:
    """Create an empty 4×4 board with two initial random tiles."""
    board: List[List[int]] = [[0] * 4 for _ in range(4)]
    board = _spawn_tile(board, rng)
    board = _spawn_tile(board, rng)
    return board


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Game2048Env:
    """Pure-Python gym-style 2048 environment (no Playwright, no browser).

    This is the **Env layer** of the Env / Train / Eval / Play stack.  It
    encapsulates all game dynamics so that the Train and Eval layers interact
    with a clean ``(reset, step, valid_actions)`` interface.

    Interface
    ---------
    The API follows the Gymnasium ``Env`` convention without depending on
    Gymnasium itself:

    * :meth:`reset` starts a new episode and returns the initial observation.
    * :meth:`step` applies an action and returns
      ``(obs, reward, terminated, truncated, info)``.
    * :meth:`valid_actions` returns indices of board-changing moves.

    Observation space
    -----------------
    A 256-dim ``float32`` one-hot vector (16 cells × 16 tile-level slots).

    Action space
    ------------
    Integer in ``{0, 1, 2, 3}`` mapping to
    ``DIRECTIONS = ["up", "down", "left", "right"]``.

    Reward shaping
    --------------
    ``log₂(merge_score + 1) + 0.1 × empty_cells``

    An *invalid* move (one that does not change the board) gives a small
    ``−0.1`` penalty without advancing the episode.

    Parameters
    ----------
    seed:
        Seed for the internal NumPy RNG.  Pass an integer for reproducible
        episodes.
    """

    #: State dimension: 256 = 16 cells × 16 one-hot levels.
    observation_size: int = _OBS_SIZE

    #: Number of discrete actions (one per direction).
    n_actions: int = 4

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._board: List[List[int]] = [[0] * 4 for _ in range(4)]
        self._score: int = 0
        self._n_steps: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Start a new episode.

        Parameters
        ----------
        seed:
            If provided, re-seed the internal RNG for this episode.

        Returns
        -------
        observation : np.ndarray
            One-hot encoded initial board state, shape ``(256,)``.
        info : dict
            ``{"board": ..., "score": 0, "max_tile": <int>, "n_steps": 0}``
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._board = _init_board(self._rng)
        self._score = 0
        self._n_steps = 0
        self._done = False
        return _encode_onehot(self._board), self._make_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply *action* to the current board.

        Parameters
        ----------
        action : int
            Direction index: 0 = up, 1 = down, 2 = left, 3 = right.

        Returns
        -------
        observation : np.ndarray   Shape ``(256,)``.
        reward      : float
        terminated  : bool         ``True`` when the game is over.
        truncated   : bool         Always ``False`` (no step-count limit).
        info        : dict

        Raises
        ------
        RuntimeError
            If called on an already-finished episode (call :meth:`reset`
            first).
        """
        if self._done:
            raise RuntimeError(
                "Cannot call step() on a finished episode; call reset() first."
            )

        direction = DIRECTIONS[action]
        new_board, merge_score = simulate_move(self._board, direction)

        # Invalid move — board unchanged; apply a small penalty.
        if _boards_equal(self._board, new_board):
            return (
                _encode_onehot(self._board),
                -0.1,
                False,
                False,
                self._make_info(),
            )

        # Valid move: spawn a new tile and update state.
        self._board = _spawn_tile(new_board, self._rng)
        self._score += merge_score
        self._n_steps += 1

        reward = math.log2(merge_score + 1) + 0.1 * sum(
            1 for r in range(4) for c in range(4) if self._board[r][c] == 0
        )

        terminated = len(self.valid_actions()) == 0
        self._done = terminated
        return (
            _encode_onehot(self._board),
            reward,
            terminated,
            False,
            self._make_info(),
        )

    def valid_actions(self) -> List[int]:
        """Return indices of actions that change the board."""
        return [
            i
            for i, d in enumerate(DIRECTIONS)
            if not _boards_equal(
                self._board, simulate_move(self._board, d)[0]
            )
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def board(self) -> List[List[int]]:
        """A copy of the current 4×4 board."""
        return [row[:] for row in self._board]

    @property
    def score(self) -> int:
        """Cumulative merge score for the current episode."""
        return self._score

    @property
    def max_tile(self) -> int:
        """Maximum tile value on the current board."""
        return max(
            self._board[r][c] for r in range(4) for c in range(4)
        )

    @property
    def n_steps(self) -> int:
        """Number of *valid* moves taken in the current episode."""
        return self._n_steps

    @property
    def is_done(self) -> bool:
        """Whether the episode has ended."""
        return self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_info(self) -> dict:
        return {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": self.max_tile,
            "n_steps": self._n_steps,
        }
