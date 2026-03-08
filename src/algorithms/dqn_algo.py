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
# DQN algorithm – version 1 (standard DQN)
# ---------------------------------------------------------------------------

class DQNAlgorithmV1(BaseAlgorithm):
    """Deep Q-Network – version 1 (standard DQN with vanilla Bellman backup).

    Uses a single target-network forward pass to compute the TD target:
    ``Q_target(s, a) = r + γ · max_a' Q_target(s', a') · (1 − done)``.
    This can overestimate Q-values; see :class:`DQNAlgorithmV2` for the
    Double-DQN fix.

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

    name = "DQN-v1"
    version = "v1"

    def __init__(
        self,
        hidden_size: int = 128,
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

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_board: Optional[List[List[int]]] = None

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction using the ε-greedy DQN policy."""
        state = _encode_board(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

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

        if self._rng.random() < self._epsilon:
            chosen_dir = self._rng.choice(valid_dirs)
        else:
            q_vals, *_ = self._q_net.forward(state)
            valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
            best = valid_idx[int(np.argmax(q_vals[valid_idx]))]
            chosen_dir = DIRECTIONS[best]

        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)
        self._prev_state = state
        self._prev_action = _DIR_INDEX[chosen_dir]
        self._prev_board = [row[:] for row in board]
        self._step += 1

        if self._step % self._target_update_freq == 0:
            self._target_net.copy_weights(self._q_net)

        return chosen_dir

    def _train_step(self) -> None:
        """Standard DQN Bellman update using the target network."""
        batch = self._rng.sample(list(self._buffer), self._batch_size)
        S = np.array([e[0] for e in batch], dtype=np.float32)
        A = np.array([e[1] for e in batch], dtype=np.int32)
        R = np.array([e[2] for e in batch], dtype=np.float32)
        S2 = np.array([e[3] for e in batch], dtype=np.float32)
        D = np.array([e[4] for e in batch], dtype=np.float32)

        # Standard DQN: max Q from target network.
        next_q, *_ = self._target_net.forward(S2)
        target_q = R + self._gamma * np.max(next_q, axis=1) * (1.0 - D)

        q_out, h2, z2, h1, z1 = self._q_net.forward(S)
        td = q_out[np.arange(self._batch_size), A] - target_q
        dL_dq = np.zeros_like(q_out)
        dL_dq[np.arange(self._batch_size), A] = (2.0 / self._batch_size) * td

        dW3 = h2.T @ dL_dq
        db3 = dL_dq.sum(axis=0)
        dh2 = dL_dq @ self._q_net.W3.T
        dz2 = dh2 * _relu_grad(z2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self._q_net.W2.T
        dz1 = dh1 * _relu_grad(z1)
        dW1 = S.T @ dz1
        db1 = dz1.sum(axis=0)

        self._q_net.sgd_update(dW1, db1, dW2, db2, dW3, db3, self._lr)


# ---------------------------------------------------------------------------
# DQN algorithm – version 2 (Double DQN)
# ---------------------------------------------------------------------------

class DQNAlgorithmV2(BaseAlgorithm):
    """Deep Q-Network – version 2 (Double DQN).

    Improvement over :class:`DQNAlgorithmV1`: uses the *online* network to
    select the greedy next action and the *target* network to evaluate its
    Q-value, which reduces the overestimation bias of standard DQN.

    Parameters
    ----------
    hidden_size:
        Number of units in each hidden layer (default 256, wider than v1).
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

    name = "DQN-v2"
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


# ---------------------------------------------------------------------------
# Backward-compatible alias — DQNAlgorithm always points to the latest version.
# ---------------------------------------------------------------------------

DQNAlgorithm = DQNAlgorithmV2


# ---------------------------------------------------------------------------
# V3 helpers: one-hot encoding, score-based reward, Adam optimizer
# ---------------------------------------------------------------------------

#: Number of tile levels per cell for the one-hot encoding.
#: Level 0 = empty; levels 1–15 = tiles 2¹ through 2¹⁵ (2 … 32 768).
_N_LEVELS = 16
#: Total dimension of the one-hot state vector (16 cells × 16 levels).
_N_STATE_V3 = 16 * _N_LEVELS  # 256


def _encode_board_onehot(board: List[List[int]]) -> np.ndarray:
    """One-hot encode the board state as a 256-dim float32 vector.

    Each of the 16 cells is represented by a one-hot vector of length
    ``_N_LEVELS``.  The 16 one-hot vectors are concatenated to give a
    256-dim output that lets the network learn independent weights for
    every (cell-position, tile-value) combination.
    """
    out = np.zeros(_N_STATE_V3, dtype=np.float32)
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            cell_idx = r * 4 + c
            level = 0 if v == 0 else min(int(math.log2(v)), _N_LEVELS - 1)
            out[cell_idx * _N_LEVELS + level] = 1.0
    return out


def _score_reward(
    prev_board: List[List[int]],
    action_idx: int,
    curr_board: List[List[int]],
) -> float:
    """Compute a shaped reward based on the actual merge score.

    This fixes the broken ``Δ(sum log₂ tiles)`` reward used in V1/V2, which
    *penalises* high-value merges.  The new reward has two components:

    * ``log₂(merge_score + 1)``: proportional to the tile value merged (0
      for non-merging moves, up to ~11 for a 1 024+1 024→2 048 merge).
    * ``0.1 · empty_count``: small bonus for keeping the board open, which
      encourages the agent to avoid filling the board prematurely.

    Both components are always ≥ 0, so the agent is never punished for
    making good merges.

    Parameters
    ----------
    prev_board:
        The board state *before* the agent's last action.
    action_idx:
        Integer action index (0–3, matching ``DIRECTIONS``).
    curr_board:
        The board state *after* the action and the new random tile.
    """
    _, score = simulate_move(prev_board, DIRECTIONS[action_idx])
    empty = sum(1 for r in range(4) for c in range(4) if curr_board[r][c] == 0)
    return math.log2(score + 1) + 0.1 * empty


class _Adam:
    """Adam optimizer for pure-NumPy parameter arrays.

    Parameters
    ----------
    lr:
        Learning rate (step size).
    beta1:
        Exponential decay rate for the first-moment estimate.
    beta2:
        Exponential decay rate for the second-moment estimate.
    eps:
        Small constant for numerical stability.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._t: int = 0
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def step(self, updates: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
        """Apply one Adam update step.

        Parameters
        ----------
        updates:
            List of ``(name, param, grad)`` triples.  Each *param* is
            updated **in-place**.  *name* is the key for the moment
            accumulators (must be unique per parameter tensor).
        """
        self._t += 1
        lr_t = (
            self.lr
            * math.sqrt(1.0 - self._beta2 ** self._t)
            / (1.0 - self._beta1 ** self._t)
        )
        for name, param, grad in updates:
            if name not in self._m:
                self._m[name] = np.zeros_like(grad)
                self._v[name] = np.zeros_like(grad)
            self._m[name] = self._beta1 * self._m[name] + (1.0 - self._beta1) * grad
            self._v[name] = self._beta2 * self._v[name] + (1.0 - self._beta2) * grad ** 2
            param -= lr_t * self._m[name] / (np.sqrt(self._v[name]) + self._eps)


# ---------------------------------------------------------------------------
# DQN algorithm – version 3 (Double DQN + Adam + one-hot state + score reward)
# ---------------------------------------------------------------------------

class DQNAlgorithmV3(BaseAlgorithm):
    """Deep Q-Network – version 3.

    This version fixes all major shortcomings of V1/V2:

    * **One-hot state encoding** (256-dim): each of the 16 cells is encoded
      as a 16-level one-hot vector so the network can learn independent
      weights for every ``(cell-position, tile-value)`` pair.
    * **Score-based reward**: ``log₂(merge_score+1) + 0.1·empty_tiles``
      instead of the V1/V2 heuristic-delta which *penalised* high-value
      merges and caused the agent to score below Random.
    * **Adam optimizer**: replaces vanilla SGD for faster, more stable
      convergence on the noisy RL loss landscape.
    * **Train every 4 steps**: reduces update correlation and computational
      overhead.
    * **Game-boundary reset** via :meth:`on_game_start`: prevents corrupt
      cross-game transitions from polluting the replay buffer.
    * Larger replay buffer (50 000) and mini-batch (128).
    * Slower ε-decay (0.9998) for adequate exploration.

    Parameters
    ----------
    hidden_size:
        Number of units in each hidden layer.
    lr:
        Adam learning rate.
    gamma:
        Discount factor for future rewards.
    epsilon_start:
        Initial ε for ε-greedy exploration.
    epsilon_end:
        Minimum ε; exploration is never fully eliminated.
    epsilon_decay:
        Multiplicative decay applied to ε after each step.
    buffer_size:
        Maximum replay buffer size.
    batch_size:
        Mini-batch size for each training step.
    target_update_freq:
        Steps between target-network weight copies.
    train_freq:
        Number of environment steps between training updates.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "DQN-v3"
    version = "v3"

    def __init__(
        self,
        hidden_size: int = 256,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9998,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        target_update_freq: int = 500,
        train_freq: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        np_seed = seed if seed is not None else 42
        self._np_rng = np.random.default_rng(np_seed)

        self._q_net = _QNetwork(_N_STATE_V3, hidden_size, _N_ACTIONS, self._np_rng)
        self._target_net = _QNetwork(_N_STATE_V3, hidden_size, _N_ACTIONS, self._np_rng)
        self._target_net.copy_weights(self._q_net)

        self._optimizer = _Adam(lr=lr)
        self._gamma = gamma
        self._epsilon = float(epsilon_start)
        self._epsilon_end = float(epsilon_end)
        self._epsilon_decay = float(epsilon_decay)
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._train_freq = train_freq

        self._buffer: deque = deque(maxlen=buffer_size)
        self._step: int = 0

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_board: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # BaseAlgorithm interface
    # ------------------------------------------------------------------

    def on_game_start(self) -> None:
        """Reset per-game transition state so games don't bleed into each other."""
        self._prev_state = None
        self._prev_action = None
        self._prev_board = None

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction using the ε-greedy DQN-v3 policy."""
        state = _encode_board_onehot(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        is_terminal = len(valid_dirs) == 0
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

        # ----------------------------------------------------------------
        # Store transition and (optionally) train
        # ----------------------------------------------------------------
        if self._prev_state is not None:
            reward = _score_reward(self._prev_board, self._prev_action, board)  # type: ignore[arg-type]
            done = float(is_terminal)
            self._buffer.append((
                self._prev_state,
                self._prev_action,
                reward,
                state.copy(),
                done,
            ))
            if (
                len(self._buffer) >= self._batch_size
                and self._step % self._train_freq == 0
            ):
                self._train_step()

        # ----------------------------------------------------------------
        # ε-greedy action selection (restricted to valid moves)
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
        """Double-DQN Bellman update with Adam optimizer."""
        buf_list = list(self._buffer)
        indices = self._np_rng.choice(len(buf_list), size=self._batch_size, replace=False)
        batch = [buf_list[i] for i in indices]

        S = np.array([e[0] for e in batch], dtype=np.float32)   # (B, 256)
        A = np.array([e[1] for e in batch], dtype=np.int32)      # (B,)
        R = np.array([e[2] for e in batch], dtype=np.float32)    # (B,)
        S2 = np.array([e[3] for e in batch], dtype=np.float32)   # (B, 256)
        D = np.array([e[4] for e in batch], dtype=np.float32)    # (B,)

        # Double DQN: online net selects next action, target net evaluates.
        online_next_q, *_ = self._q_net.forward(S2)
        best_next = np.argmax(online_next_q, axis=1)              # (B,)
        target_next_q, *_ = self._target_net.forward(S2)
        target_q = (
            R
            + self._gamma
            * target_next_q[np.arange(self._batch_size), best_next]
            * (1.0 - D)
        )

        q_out, h2, z2, h1, z1 = self._q_net.forward(S)
        td = q_out[np.arange(self._batch_size), A] - target_q    # (B,)
        dL_dq = np.zeros_like(q_out)
        dL_dq[np.arange(self._batch_size), A] = (2.0 / self._batch_size) * td

        dW3 = h2.T @ dL_dq
        db3 = dL_dq.sum(axis=0)
        dh2 = dL_dq @ self._q_net.W3.T
        dz2 = dh2 * _relu_grad(z2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self._q_net.W2.T
        dz1 = dh1 * _relu_grad(z1)
        dW1 = S.T @ dz1
        db1 = dz1.sum(axis=0)

        self._optimizer.step([
            ("W1", self._q_net.W1, dW1),
            ("b1", self._q_net.b1, db1),
            ("W2", self._q_net.W2, dW2),
            ("b2", self._q_net.b2, db2),
            ("W3", self._q_net.W3, dW3),
            ("b3", self._q_net.b3, db3),
        ])

