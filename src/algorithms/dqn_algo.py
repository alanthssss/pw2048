"""DQN algorithm: Deep Q-Network reinforcement learning for 2048."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Optional

import numpy as np

from .base import BaseAlgorithm
from .greedy_algo import simulate_move, _boards_equal
from src.game import DIRECTIONS

# Map direction strings to integer indices (must be consistent with DIRECTIONS).
_DIR_INDEX: dict[str, int] = {d: i for i, d in enumerate(DIRECTIONS)}
_N_ACTIONS = 4
_N_STATE = 16


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def _encode_board(board: List[List[int]]) -> np.ndarray:
    """Encode the board as a 16-dim float32 vector using normalized log₂ values.

    Empty cells map to 0; tile value *v* maps to ``log2(v) / 11``, so the
    maximum tile 2048 (= 2¹¹) maps to 1.0.
    """
    flat = [board[r][c] for r in range(4) for c in range(4)]
    return np.array(
        [math.log2(v) / 11.0 if v > 0 else 0.0 for v in flat],
        dtype=np.float32,
    )


def _board_heuristic(board: List[List[int]]) -> float:
    """Simple heuristic: sum of log₂ tile values (proxy for board quality)."""
    return sum(
        math.log2(v)
        for r in range(4)
        for c in range(4)
        for v in [board[r][c]]
        if v > 0
    )


# ---------------------------------------------------------------------------
# Pure-numpy 2-layer MLP
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    """Element-wise derivative of ReLU evaluated at pre-activation *x*."""
    return (x > 0).astype(np.float32)


class _QNetwork:
    """Two-layer fully-connected Q-network implemented in pure NumPy.

    Architecture: ``in_dim → hidden → hidden → out_dim``

    The first two linear transforms use ReLU activations; the output layer is
    linear (no activation) to allow unbounded Q-values.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        rng: np.random.Generator,
    ) -> None:
        # He (Kaiming) initialisation for layers preceding ReLU.
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, s1, (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, s2, (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, s2, (hidden, out_dim)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input of shape ``(batch, in_dim)`` or ``(in_dim,)`` for a
            single sample.

        Returns
        -------
        tuple
            ``(q_values, h2, z2, h1, z1)`` where *z_i* are pre-activations
            and *h_i* are post-activations (needed for backprop).
        """
        z1 = x @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _relu(z2)
        q = h2 @ self.W3 + self.b3
        return q, h2, z2, h1, z1

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def copy_weights(self, other: "_QNetwork") -> None:
        """Copy *other*'s weights into this network (for target-net sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    # ------------------------------------------------------------------
    # Gradient-descent update
    # ------------------------------------------------------------------

    def sgd_update(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
        dW3: np.ndarray,
        db3: np.ndarray,
        lr: float,
    ) -> None:
        """Apply a single SGD step."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


# ---------------------------------------------------------------------------
# DQN algorithm
# ---------------------------------------------------------------------------

class DQNAlgorithm(BaseAlgorithm):
    """Deep Q-Network (DQN) reinforcement-learning algorithm for 2048.

    The agent maintains an online Q-network and a lagged target network.
    Experiences ``(state, action, reward, next_state, done)`` are stored in a
    replay buffer and sampled uniformly for mini-batch Q-learning updates.

    The board state is encoded as a 16-dimensional vector of normalised log₂
    tile values.  Actions are masked to valid moves at inference time.

    Parameters
    ----------
    hidden_size:
        Number of units in each hidden layer.
    lr:
        Learning rate for SGD updates.
    gamma:
        Discount factor for future rewards.
    epsilon_start:
        Initial ε for ε-greedy exploration.
    epsilon_end:
        Minimum ε; exploration is never fully eliminated.
    epsilon_decay:
        Multiplicative decay applied to ε after each step.
    buffer_size:
        Maximum number of transitions in the replay buffer.
    batch_size:
        Number of transitions sampled per training step.
    target_update_freq:
        Number of steps between target-network weight copies.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "DQN"
    version = "v2"

    def __init__(
        self,
        hidden_size: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        np_seed = seed if seed is not None else 42
        self._np_rng = np.random.default_rng(np_seed)

        self._q_net = _QNetwork(_N_STATE, hidden_size, _N_ACTIONS, self._np_rng)
        self._target_net = _QNetwork(_N_STATE, hidden_size, _N_ACTIONS, self._np_rng)
        self._target_net.copy_weights(self._q_net)

        self._lr = lr
        self._gamma = gamma
        self._epsilon = float(epsilon_start)
        self._epsilon_end = float(epsilon_end)
        self._epsilon_decay = float(epsilon_decay)
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq

        self._buffer: deque = deque(maxlen=buffer_size)
        self._step: int = 0

        # State from the previous call to choose_move (for building transitions).
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_board: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction using the ε-greedy DQN policy.

        On each call the agent:

        1. Stores the transition from the previous step (if any) and trains
           when the replay buffer is large enough.
        2. Selects an action via ε-greedy, restricted to valid moves.
        3. Decays ε and periodically syncs the target network.
        """
        state = _encode_board(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

        # ----------------------------------------------------------------
        # Store transition and learn
        # ----------------------------------------------------------------
        if self._prev_state is not None:
            reward = _board_heuristic(board) - _board_heuristic(self._prev_board)  # type: ignore[arg-type]
            done = len(valid_dirs) == 0
            self._buffer.append((
                self._prev_state,
                self._prev_action,
                float(reward),
                state.copy(),
                float(done),
            ))
            if len(self._buffer) >= self._batch_size:
                self._train_step()

        # ----------------------------------------------------------------
        # ε-greedy action selection (only valid moves)
        # ----------------------------------------------------------------
        if self._rng.random() < self._epsilon:
            chosen_dir = self._rng.choice(valid_dirs)
        else:
            q_vals, *_ = self._q_net.forward(state)
            valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
            best = valid_idx[int(np.argmax(q_vals[valid_idx]))]
            chosen_dir = DIRECTIONS[best]

        # ----------------------------------------------------------------
        # Housekeeping
        # ----------------------------------------------------------------
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)
        self._prev_state = state
        self._prev_action = _DIR_INDEX[chosen_dir]
        self._prev_board = [row[:] for row in board]
        self._step += 1

        if self._step % self._target_update_freq == 0:
            self._target_net.copy_weights(self._q_net)

        return chosen_dir

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_step(self) -> None:
        """Sample a mini-batch from the replay buffer and update the Q-network.

        Uses Double DQN: the online network selects the greedy next action and
        the target network evaluates its Q-value.  This reduces the
        overestimation bias present in vanilla DQN.
        """
        batch = self._rng.sample(list(self._buffer), self._batch_size)
        S = np.array([e[0] for e in batch], dtype=np.float32)   # (B, 16)
        A = np.array([e[1] for e in batch], dtype=np.int32)      # (B,)
        R = np.array([e[2] for e in batch], dtype=np.float32)    # (B,)
        S2 = np.array([e[3] for e in batch], dtype=np.float32)   # (B, 16)
        D = np.array([e[4] for e in batch], dtype=np.float32)    # (B,)

        # Double DQN: online net picks the greedy action for next state,
        # target net evaluates that action's Q-value.
        online_next_q, *_ = self._q_net.forward(S2)
        best_next_actions = np.argmax(online_next_q, axis=1)     # (B,)
        target_next_q, *_ = self._target_net.forward(S2)
        target_q = (
            R
            + self._gamma
            * target_next_q[np.arange(self._batch_size), best_next_actions]
            * (1.0 - D)
        )

        # Forward pass through the online Q-network.
        q_out, h2, z2, h1, z1 = self._q_net.forward(S)

        # MSE loss gradient w.r.t. Q(s, a) for selected actions only.
        td = q_out[np.arange(self._batch_size), A] - target_q      # (B,)
        dL_dq = np.zeros_like(q_out)                                # (B, 4)
        dL_dq[np.arange(self._batch_size), A] = (2.0 / self._batch_size) * td

        # Backprop through W3 / b3.
        dW3 = h2.T @ dL_dq                          # (hidden, 4)
        db3 = dL_dq.sum(axis=0)                     # (4,)

        # Backprop through ReLU (layer 2) → W2 / b2.
        dh2 = dL_dq @ self._q_net.W3.T             # (B, hidden)
        dz2 = dh2 * _relu_grad(z2)                 # (B, hidden)
        dW2 = h1.T @ dz2                            # (hidden, hidden)
        db2 = dz2.sum(axis=0)                       # (hidden,)

        # Backprop through ReLU (layer 1) → W1 / b1.
        dh1 = dz2 @ self._q_net.W2.T               # (B, hidden)
        dz1 = dh1 * _relu_grad(z1)                 # (B, hidden)
        dW1 = S.T @ dz1                             # (16, hidden)
        db1 = dz1.sum(axis=0)                       # (hidden,)

        self._q_net.sgd_update(dW1, db1, dW2, db2, dW3, db3, self._lr)
