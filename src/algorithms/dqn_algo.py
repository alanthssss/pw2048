"""DQN algorithm: Deep Q-Network reinforcement learning for 2048."""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .base import BaseAlgorithm
from .greedy_algo import simulate_move, _boards_equal
from src.game import DIRECTIONS

# ---------------------------------------------------------------------------
# Optional PyTorch backend (GPU / MPS / CUDA acceleration)
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _detect_device() -> Optional[str]:
    """Return the best available compute device string for PyTorch.

    Detection priority: Apple Silicon MPS → CUDA GPU → CPU.  Returns ``None``
    if PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        return None
    if _torch.backends.mps.is_available():
        return "mps"
    if _torch.cuda.is_available():
        return "cuda"
    return "cpu"

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
# PyTorch Q-network (optional — only instantiated when torch is available)
# ---------------------------------------------------------------------------

class _TorchQNetwork(_nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Two-layer Q-network implemented with PyTorch for GPU/MPS acceleration.

    Architecture matches :class:`_QNetwork`: ``in_dim → hidden → hidden → out_dim``
    with ReLU activations on the first two layers and a linear output layer.

    Parameters
    ----------
    in_dim:
        Input dimensionality (256 for the one-hot board encoding).
    hidden:
        Hidden layer width.
    out_dim:
        Output dimensionality (4 Q-values, one per action).
    device:
        Target compute device string (``"mps"``, ``"cuda"``, or ``"cpu"``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        device: str,
    ) -> None:
        super().__init__()
        self.fc1 = _nn.Linear(in_dim, hidden)
        self.fc2 = _nn.Linear(hidden, hidden)
        self.fc3 = _nn.Linear(hidden, out_dim)
        self._device_str = device
        # He (Kaiming) initialisation for layers preceding ReLU.
        _nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        _nn.init.zeros_(self.fc1.bias)
        _nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        _nn.init.zeros_(self.fc2.bias)
        _nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        _nn.init.zeros_(self.fc3.bias)
        self.to(_torch.device(device))

    def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":  # type: ignore[override]
        """Forward pass for a batch of states."""
        x = _F.relu(self.fc1(x))
        x = _F.relu(self.fc2(x))
        return self.fc3(x)

    def copy_weights(self, other: "_TorchQNetwork") -> None:
        """Copy *other*'s weights into this network."""
        self.load_state_dict({k: v.clone() for k, v in other.state_dict().items()})

    def weights_to_numpy(self) -> dict[str, np.ndarray]:
        """Return weights as a numpy dict using the same keys as :class:`_QNetwork`."""
        sd = self.state_dict()
        return {
            # Transpose: torch stores (out, in); numpy convention is (in, out).
            "W1": sd["fc1.weight"].cpu().numpy().T,
            "b1": sd["fc1.bias"].cpu().numpy(),
            "W2": sd["fc2.weight"].cpu().numpy().T,
            "b2": sd["fc2.bias"].cpu().numpy(),
            "W3": sd["fc3.weight"].cpu().numpy().T,
            "b3": sd["fc3.bias"].cpu().numpy(),
        }

    def weights_from_numpy(self, d: dict, prefix: str = "") -> None:
        """Load weights from a numpy dict (inverse of :meth:`weights_to_numpy`)."""
        device = _torch.device(self._device_str)
        new_sd = {
            "fc1.weight": _torch.from_numpy(d[f"{prefix}W1"].T.copy()).to(device),
            "fc1.bias":   _torch.from_numpy(d[f"{prefix}b1"].copy()).to(device),
            "fc2.weight": _torch.from_numpy(d[f"{prefix}W2"].T.copy()).to(device),
            "fc2.bias":   _torch.from_numpy(d[f"{prefix}b2"].copy()).to(device),
            "fc3.weight": _torch.from_numpy(d[f"{prefix}W3"].T.copy()).to(device),
            "fc3.bias":   _torch.from_numpy(d[f"{prefix}b3"].copy()).to(device),
        }
        self.load_state_dict(new_sd)


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
# Shared simulation helpers (browser-free in-process game simulation)
# ---------------------------------------------------------------------------

def _spawn_tile_np(
    board: List[List[int]], rng: np.random.Generator
) -> List[List[int]]:
    """Return a copy of *board* with one new tile spawned at a random empty cell.

    Follows the 2048 spawn distribution: 90 % → 2, 10 % → 4.  If the board
    has no empty cells the board is returned unchanged.
    """
    empty = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empty:
        return [row[:] for row in board]
    new_board = [row[:] for row in board]
    idx = int(rng.integers(len(empty)))
    r, c = empty[idx]
    new_board[r][c] = 2 if rng.random() < 0.9 else 4
    return new_board


def _init_board_np(rng: np.random.Generator) -> List[List[int]]:
    """Create an empty 4×4 board with two initial tiles (standard 2048 start)."""
    board: List[List[int]] = [[0] * 4 for _ in range(4)]
    board = _spawn_tile_np(board, rng)
    board = _spawn_tile_np(board, rng)
    return board


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
    * **Behavioural-cloning pre-training**: before any RL experience the
      Q-network is warmed up by imitating the Heuristic algorithm on
      ``n_pretrain_games`` simulated games (browser-free, in-process).
      After pre-training the agent already plays at heuristic level, so RL
      fine-tuning can improve further rather than spending most of a run
      re-discovering basic strategy.
    * Larger replay buffer (50 000) and mini-batch (128).
    * Slower ε-decay (0.9998) for adequate exploration.
    * **Optional GPU acceleration**: when PyTorch is installed the network runs
      on Apple Silicon MPS, CUDA, or CPU using hardware-accelerated matrix ops.
      Pass ``device="mps"`` / ``"cuda"`` / ``"cpu"`` to override auto-detection,
      or ``device="numpy"`` to force the pure-NumPy backend regardless.

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
    n_pretrain_games:
        Number of in-process heuristic games used for behavioural-cloning
        pre-training at construction time.  Set to 0 to disable.  The
        default of 50 games (≈ 45 000 training samples) completes in about
        4–5 seconds and gives the network a solid heuristic baseline.
    seed:
        Optional RNG seed for reproducibility.
    device:
        Compute device for the neural network.  ``None`` (default) enables
        automatic detection: PyTorch MPS → CUDA → CPU → NumPy fallback.
        Pass ``"numpy"`` to force the pure-NumPy backend, or any valid
        ``torch.device`` string (``"mps"``, ``"cuda"``, ``"cpu"``) to pin a
        specific PyTorch device.
    """

    name = "DQN-v3"
    version = "v3"
    #: Indicates this algorithm supports save/load checkpoint.
    supports_checkpoint: bool = True

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
        n_pretrain_games: int = 50,
        seed: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self._rng = random.Random(seed)
        np_seed = seed if seed is not None else 42
        self._np_rng = np.random.default_rng(np_seed)

        # ------------------------------------------------------------------
        # Choose compute backend: PyTorch (GPU/MPS/CPU) or pure NumPy.
        # ------------------------------------------------------------------
        if device == "numpy":
            self._torch_device: Optional[str] = None
        elif device is not None:
            # Explicit device string requested — use torch on that device.
            self._torch_device = device if _TORCH_AVAILABLE else None
        else:
            # Auto-detect best available backend.
            self._torch_device = _detect_device()  # None if torch not installed

        self._use_torch: bool = self._torch_device is not None

        if self._use_torch:
            self._q_net: Union[_QNetwork, "_TorchQNetwork"] = _TorchQNetwork(
                _N_STATE_V3, hidden_size, _N_ACTIONS, self._torch_device  # type: ignore[arg-type]
            )
            self._target_net: Union[_QNetwork, "_TorchQNetwork"] = _TorchQNetwork(
                _N_STATE_V3, hidden_size, _N_ACTIONS, self._torch_device  # type: ignore[arg-type]
            )
            self._target_net.copy_weights(self._q_net)  # type: ignore[arg-type]
            self._torch_optimizer = _torch.optim.Adam(  # type: ignore[union-attr]
                self._q_net.parameters(),  # type: ignore[union-attr]
                lr=lr,
            )
            self._optimizer = _Adam(lr=lr)  # used only for the NumPy-format checkpoint adam state (not for torch training)
        else:
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

        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        if ckpt is not None and ckpt.exists():
            # Load saved weights — skip BC pre-training entirely.
            self.load_checkpoint(ckpt)
        elif n_pretrain_games > 0:
            self._pretrain_bc(n_pretrain_games)

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
            q_vals = self._forward_q(state)
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

    def predict(self, board: List[List[int]]) -> str:
        """Return the greedy action without any exploration or buffer update.

        Used by :class:`src.rl_trainer.EvalCallback` for deterministic
        evaluation episodes.  Never touches the replay buffer, ε, or the step
        counter.

        Parameters
        ----------
        board : list[list[int]]
            Current 4×4 board.

        Returns
        -------
        str
            The greedy direction selected by the Q-network (argmax over valid
            actions, no ε-greedy randomness).
        """
        state = _encode_board_onehot(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)
        q_vals = self._forward_q(state)
        valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
        best = valid_idx[int(np.argmax(q_vals[valid_idx]))]
        return DIRECTIONS[best]

    # ------------------------------------------------------------------
    # Compute-backend helpers
    # ------------------------------------------------------------------

    def _forward_q(self, state: np.ndarray) -> np.ndarray:
        """Single-state forward pass of the online Q-network → numpy Q-values."""
        if self._use_torch:
            with _torch.no_grad():
                t = _torch.from_numpy(state).unsqueeze(0).to(self._torch_device)
                return self._q_net(t).squeeze(0).cpu().numpy()  # type: ignore[union-attr]
        else:
            q_vals, *_ = self._q_net.forward(state)  # type: ignore[union-attr]
            return q_vals

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_step(self) -> None:
        """Dispatch to torch or numpy training backend."""
        if self._use_torch:
            self._train_step_torch()
        else:
            self._train_step_numpy()

    def _train_step_numpy(self) -> None:
        """Double-DQN Bellman update with NumPy Adam optimizer."""
        buf_list = list(self._buffer)
        indices = self._np_rng.choice(len(buf_list), size=self._batch_size, replace=False)
        batch = [buf_list[i] for i in indices]

        S = np.array([e[0] for e in batch], dtype=np.float32)   # (B, 256)
        A = np.array([e[1] for e in batch], dtype=np.int32)      # (B,)
        R = np.array([e[2] for e in batch], dtype=np.float32)    # (B,)
        S2 = np.array([e[3] for e in batch], dtype=np.float32)   # (B, 256)
        D = np.array([e[4] for e in batch], dtype=np.float32)    # (B,)

        # Double DQN: online net selects next action, target net evaluates.
        online_next_q, *_ = self._q_net.forward(S2)  # type: ignore[union-attr]
        best_next = np.argmax(online_next_q, axis=1)              # (B,)
        target_next_q, *_ = self._target_net.forward(S2)  # type: ignore[union-attr]
        target_q = (
            R
            + self._gamma
            * target_next_q[np.arange(self._batch_size), best_next]
            * (1.0 - D)
        )

        q_out, h2, z2, h1, z1 = self._q_net.forward(S)  # type: ignore[union-attr]
        td = q_out[np.arange(self._batch_size), A] - target_q    # (B,)
        dL_dq = np.zeros_like(q_out)
        dL_dq[np.arange(self._batch_size), A] = (2.0 / self._batch_size) * td

        dW3 = h2.T @ dL_dq
        db3 = dL_dq.sum(axis=0)
        dh2 = dL_dq @ self._q_net.W3.T  # type: ignore[union-attr]
        dz2 = dh2 * _relu_grad(z2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self._q_net.W2.T  # type: ignore[union-attr]
        dz1 = dh1 * _relu_grad(z1)
        dW1 = S.T @ dz1
        db1 = dz1.sum(axis=0)

        self._optimizer.step([
            ("W1", self._q_net.W1, dW1),  # type: ignore[union-attr]
            ("b1", self._q_net.b1, db1),  # type: ignore[union-attr]
            ("W2", self._q_net.W2, dW2),  # type: ignore[union-attr]
            ("b2", self._q_net.b2, db2),  # type: ignore[union-attr]
            ("W3", self._q_net.W3, dW3),  # type: ignore[union-attr]
            ("b3", self._q_net.b3, db3),  # type: ignore[union-attr]
        ])

    def _train_step_torch(self) -> None:
        """Double-DQN Bellman update using PyTorch autograd (GPU/MPS/CPU)."""
        buf_list = list(self._buffer)
        indices = self._np_rng.choice(len(buf_list), size=self._batch_size, replace=False)
        batch = [buf_list[i] for i in indices]

        device = _torch.device(self._torch_device)  # type: ignore[arg-type]
        S  = _torch.from_numpy(np.array([e[0] for e in batch], dtype=np.float32)).to(device)
        # Actions stored as int in the buffer; build int32 array then cast to int64 on device.
        A  = _torch.from_numpy(np.array([e[1] for e in batch], dtype=np.int32)).to(device=device, dtype=_torch.int64)
        R  = _torch.from_numpy(np.array([e[2] for e in batch], dtype=np.float32)).to(device)
        S2 = _torch.from_numpy(np.array([e[3] for e in batch], dtype=np.float32)).to(device)
        D  = _torch.from_numpy(np.array([e[4] for e in batch], dtype=np.float32)).to(device)

        # Double DQN: online net picks next action, target net evaluates it.
        with _torch.no_grad():
            online_next_q = self._q_net(S2)  # type: ignore[operator]
            best_next = online_next_q.argmax(dim=1)
            target_next_q = self._target_net(S2)  # type: ignore[operator]
            target_q = R + self._gamma * target_next_q[_torch.arange(len(S)), best_next] * (1.0 - D)

        q_out = self._q_net(S)  # type: ignore[operator]
        q_pred = q_out[_torch.arange(len(S)), A]
        loss = ((q_pred - target_q) ** 2).mean()

        self._torch_optimizer.zero_grad()
        loss.backward()
        self._torch_optimizer.step()

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist all trainable state to a ``.npz`` file.

        Saved state
        -----------
        * Q-network and target-network weights (in NumPy format for backend
          portability — checkpoints created with the PyTorch backend can be
          loaded by the NumPy backend and vice versa).
        * Adam optimizer state (step counter, first and second moment vectors);
          **only saved when using the NumPy backend** — the PyTorch Adam
          optimizer state is not serialised, so it resets when the checkpoint
          is loaded.  Network weights (the critical state) are always preserved.
        * Current ε (exploration rate) and global step counter

        The replay buffer is intentionally **not** saved — it is large (up to
        50 000 transitions) and old transitions become less relevant over time.
        The agent will refill the buffer quickly at the start of the next run.

        Parameters
        ----------
        path:
            Destination file.  A ``.npz`` extension is recommended; NumPy
            adds it automatically if omitted.
        """
        if self._use_torch:
            q_np  = self._q_net.weights_to_numpy()       # type: ignore[union-attr]
            t_np  = self._target_net.weights_to_numpy()  # type: ignore[union-attr]
            data: dict[str, np.ndarray] = {
                "q_W1": q_np["W1"], "q_b1": q_np["b1"],
                "q_W2": q_np["W2"], "q_b2": q_np["b2"],
                "q_W3": q_np["W3"], "q_b3": q_np["b3"],
                "t_W1": t_np["W1"], "t_b1": t_np["b1"],
                "t_W2": t_np["W2"], "t_b2": t_np["b2"],
                "t_W3": t_np["W3"], "t_b3": t_np["b3"],
                "epsilon": np.array([self._epsilon], dtype=np.float32),
                "step":    np.array([self._step],    dtype=np.int64),
                # No Adam state saved for torch backend (optimizer resets on load).
                "adam_t":  np.array([0], dtype=np.int64),
            }
        else:
            data = {
                # Online Q-network
                "q_W1": self._q_net.W1,  # type: ignore[union-attr]
                "q_b1": self._q_net.b1,  # type: ignore[union-attr]
                "q_W2": self._q_net.W2,  # type: ignore[union-attr]
                "q_b2": self._q_net.b2,  # type: ignore[union-attr]
                "q_W3": self._q_net.W3,  # type: ignore[union-attr]
                "q_b3": self._q_net.b3,  # type: ignore[union-attr]
                # Target network
                "t_W1": self._target_net.W1,  # type: ignore[union-attr]
                "t_b1": self._target_net.b1,  # type: ignore[union-attr]
                "t_W2": self._target_net.W2,  # type: ignore[union-attr]
                "t_b2": self._target_net.b2,  # type: ignore[union-attr]
                "t_W3": self._target_net.W3,  # type: ignore[union-attr]
                "t_b3": self._target_net.b3,  # type: ignore[union-attr]
                # Scalar training state
                "epsilon": np.array([self._epsilon], dtype=np.float32),
                "step":    np.array([self._step],    dtype=np.int64),
                # Adam step counter
                "adam_t":  np.array([self._optimizer._t], dtype=np.int64),
            }
            # Adam moment vectors (variable set of keys depending on what was trained).
            for k, v in self._optimizer._m.items():
                data[f"adam_m_{k}"] = v
            for k, v in self._optimizer._v.items():
                data[f"adam_v_{k}"] = v
        np.savez(path, **data)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore trainable state saved by :meth:`save_checkpoint`.

        This method may be called either from :meth:`__init__` (when a
        ``checkpoint_path`` is supplied) or externally.  After loading, ε is
        taken from the checkpoint so exploration continues from where the last
        run left off.

        Checkpoints are backend-portable: a checkpoint saved with the PyTorch
        backend can be loaded by the NumPy backend and vice versa.

        Parameters
        ----------
        path:
            Source ``.npz`` file written by :meth:`save_checkpoint`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        d = np.load(path)
        if self._use_torch:
            self._q_net.weights_from_numpy(      # type: ignore[union-attr]
                {"W1": d["q_W1"], "b1": d["q_b1"], "W2": d["q_W2"],
                 "b2": d["q_b2"], "W3": d["q_W3"], "b3": d["q_b3"]}
            )
            self._target_net.weights_from_numpy(  # type: ignore[union-attr]
                {"W1": d["t_W1"], "b1": d["t_b1"], "W2": d["t_W2"],
                 "b2": d["t_b2"], "W3": d["t_W3"], "b3": d["t_b3"]}
            )
        else:
            # Online Q-network
            self._q_net.W1 = d["q_W1"].copy()  # type: ignore[union-attr]
            self._q_net.b1 = d["q_b1"].copy()  # type: ignore[union-attr]
            self._q_net.W2 = d["q_W2"].copy()  # type: ignore[union-attr]
            self._q_net.b2 = d["q_b2"].copy()  # type: ignore[union-attr]
            self._q_net.W3 = d["q_W3"].copy()  # type: ignore[union-attr]
            self._q_net.b3 = d["q_b3"].copy()  # type: ignore[union-attr]
            # Target network
            self._target_net.W1 = d["t_W1"].copy()  # type: ignore[union-attr]
            self._target_net.b1 = d["t_b1"].copy()  # type: ignore[union-attr]
            self._target_net.W2 = d["t_W2"].copy()  # type: ignore[union-attr]
            self._target_net.b2 = d["t_b2"].copy()  # type: ignore[union-attr]
            self._target_net.W3 = d["t_W3"].copy()  # type: ignore[union-attr]
            self._target_net.b3 = d["t_b3"].copy()  # type: ignore[union-attr]
            # Adam optimizer
            self._optimizer._t = int(d["adam_t"][0])
            self._optimizer._m = {
                k[7:]: d[k].copy() for k in d.files if k.startswith("adam_m_")
            }
            self._optimizer._v = {
                k[7:]: d[k].copy() for k in d.files if k.startswith("adam_v_")
            }
        # Scalar state (always present).
        self._epsilon = float(d["epsilon"][0])
        self._step    = int(d["step"][0])

    # ------------------------------------------------------------------
    # Behavioural-cloning pre-training
    # ------------------------------------------------------------------

    def _pretrain_bc(self, n_games: int) -> None:
        """Warm-start the Q-network by imitating the Heuristic algorithm.

        Runs ``n_games`` 2048 games in-process (no browser) using the
        Heuristic policy to pick moves and tile-spawn simulation to advance
        board state.  Collected ``(state, heuristic_action)`` pairs are used
        for supervised cross-entropy training of the Q-network, which gives
        the agent a strong starting point before any RL experience.

        After pre-training:

        * ε is capped at ``0.3`` so the agent mostly exploits the learned
          policy rather than exploring randomly from game 1.
        * The target network is synchronised with the trained Q-network.
        """
        from .heuristic_algo import _score_board  # lazy import

        all_states: list[np.ndarray] = []
        all_actions: list[int] = []

        for _ in range(n_games):
            board = _init_board_np(self._np_rng)
            for _ in range(2_000):
                best_dir: Optional[str] = None
                best_score = float("-inf")
                for d in DIRECTIONS:
                    nb, _ = simulate_move(board, d)
                    if not _boards_equal(board, nb):
                        s = _score_board(nb)
                        if s > best_score:
                            best_score = s
                            best_dir = d
                if best_dir is None:
                    break  # game over

                all_states.append(_encode_board_onehot(board))
                all_actions.append(_DIR_INDEX[best_dir])

                board, _ = simulate_move(board, best_dir)
                board = _spawn_tile_np(board, self._np_rng)

        if not all_states:
            return

        states = np.array(all_states, dtype=np.float32)   # (N, 256)
        actions = np.array(all_actions, dtype=np.int32)    # (N,)
        n = len(states)
        bc_batch = min(256, n)

        if self._use_torch:
            # PyTorch cross-entropy supervised pass.
            device = _torch.device(self._torch_device)  # type: ignore[arg-type]
            for _epoch in range(3):
                perm = self._np_rng.permutation(n)
                for start in range(0, n - bc_batch + 1, bc_batch):
                    S_np = states[perm[start:start + bc_batch]]
                    A_np = actions[perm[start:start + bc_batch]]
                    S_t = _torch.from_numpy(S_np).to(device)
                    A_t = _torch.from_numpy(A_np.astype(np.int64)).to(device)
                    logits = self._q_net(S_t)  # type: ignore[operator]
                    loss = _F.cross_entropy(logits, A_t)
                    self._torch_optimizer.zero_grad()
                    loss.backward()
                    self._torch_optimizer.step()
        else:
            # NumPy manual backprop supervised pass (existing code).
            for _epoch in range(3):
                perm = self._np_rng.permutation(n)
                for start in range(0, n - bc_batch + 1, bc_batch):
                    S = states[perm[start:start + bc_batch]]       # (B, 256)
                    A = actions[perm[start:start + bc_batch]]      # (B,)
                    B = len(S)

                    q, h2, z2, h1, z1 = self._q_net.forward(S)  # type: ignore[union-attr]

                    # Softmax cross-entropy: dL/dq = (softmax(q) − one_hot(a)) / B
                    q_shifted = q - q.max(axis=1, keepdims=True)
                    exp_q = np.exp(q_shifted)
                    probs = exp_q / (exp_q.sum(axis=1, keepdims=True) + 1e-8)
                    dL_dq = probs.copy()
                    dL_dq[np.arange(B), A] -= 1.0
                    dL_dq /= B

                    dW3 = h2.T @ dL_dq
                    db3 = dL_dq.sum(axis=0)
                    dh2 = dL_dq @ self._q_net.W3.T  # type: ignore[union-attr]
                    dz2 = dh2 * _relu_grad(z2)
                    dW2 = h1.T @ dz2
                    db2 = dz2.sum(axis=0)
                    dh1 = dz2 @ self._q_net.W2.T  # type: ignore[union-attr]
                    dz1 = dh1 * _relu_grad(z1)
                    dW1 = S.T @ dz1
                    db1 = dz1.sum(axis=0)

                    self._optimizer.step([
                        ("W1", self._q_net.W1, dW1),  # type: ignore[union-attr]
                        ("b1", self._q_net.b1, db1),  # type: ignore[union-attr]
                        ("W2", self._q_net.W2, dW2),  # type: ignore[union-attr]
                        ("b2", self._q_net.b2, db2),  # type: ignore[union-attr]
                        ("W3", self._q_net.W3, dW3),  # type: ignore[union-attr]
                        ("b3", self._q_net.b3, db3),  # type: ignore[union-attr]
                    ])

        # Start RL with a lower ε — the policy already knows the basics.
        self._epsilon = min(self._epsilon, 0.3)
        # Keep target net in sync with the pre-trained weights.
        self._target_net.copy_weights(self._q_net)

