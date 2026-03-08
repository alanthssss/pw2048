"""PPO algorithm: Proximal Policy Optimization reinforcement learning for 2048."""

from __future__ import annotations

import math
import random
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
# Board encoding (shared with DQN)
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
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    """Element-wise derivative of ReLU evaluated at pre-activation *x*."""
    return (x > 0).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Pure-numpy actor-critic network
# ---------------------------------------------------------------------------

class _ActorCritic:
    """Shared-backbone actor-critic network implemented in pure NumPy.

    Architecture
    ------------
    Input (16) → Linear → ReLU → Linear → ReLU → actor head (4 logits)
                                                 → critic head (1 value)

    The two hidden layers share weights between the actor and the critic.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        rng: np.random.Generator,
    ) -> None:
        # He initialisation for layers preceding ReLU.
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden)
        # Shared hidden layers.
        self.W1 = rng.normal(0, s1, (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, s2, (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        # Actor head (policy logits) — small init keeps initial probs near uniform.
        self.W_a = rng.normal(0, 0.01, (hidden, _N_ACTIONS)).astype(np.float32)
        self.b_a = np.zeros(_N_ACTIONS, dtype=np.float32)
        # Critic head (state value).
        self.W_v = rng.normal(0, s2, (hidden, 1)).astype(np.float32)
        self.b_v = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass for a batch of states.

        Parameters
        ----------
        x:
            Shape ``(batch, 16)`` or ``(16,)`` for a single sample.

        Returns
        -------
        tuple
            ``(logits, values, h2, z2, h1, z1)``
            *logits*: raw policy scores ``(batch, 4)``
            *values*: state-value estimates ``(batch,)``
            *h2*, *z2*, *h1*, *z1*: intermediate activations for backprop.
        """
        z1 = x @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _relu(z2)
        logits = h2 @ self.W_a + self.b_a
        values = (h2 @ self.W_v + self.b_v).squeeze(-1)
        return logits, values, h2, z2, h1, z1

    # ------------------------------------------------------------------
    # Gradient-descent update
    # ------------------------------------------------------------------

    def sgd_update(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
        dW_a: np.ndarray,
        db_a: np.ndarray,
        dW_v: np.ndarray,
        db_v: np.ndarray,
        lr: float,
    ) -> None:
        """Apply a single SGD step."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W_a -= lr * dW_a
        self.b_a -= lr * db_a
        self.W_v -= lr * dW_v
        self.b_v -= lr * db_v


# ---------------------------------------------------------------------------
# PPO algorithm
# ---------------------------------------------------------------------------

class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization (PPO-Clip) reinforcement-learning algorithm for 2048.

    The agent collects a fixed-length rollout of ``update_freq`` steps and
    then performs ``n_epochs`` passes of PPO updates over the collected data.

    The board state is encoded as a 16-dimensional vector of normalised log₂
    tile values.  At inference time, a probability distribution is computed
    over valid moves only (invalid moves are masked with a large negative
    logit), and an action is sampled from that distribution.

    Parameters
    ----------
    hidden_size:
        Number of units in each shared hidden layer.
    lr:
        Learning rate for SGD updates.
    gamma:
        Discount factor used in GAE advantage estimation.
    lam:
        λ parameter for Generalized Advantage Estimation (GAE).  Higher
        values reduce bias at the cost of higher variance.
    clip_eps:
        PPO clipping parameter ε; constrains the policy update ratio to
        ``[1 − ε, 1 + ε]``.
    n_epochs:
        Number of gradient-update epochs over the collected rollout.
    update_freq:
        Number of steps to collect before performing a PPO update.
    entropy_coef:
        Coefficient on the entropy bonus; encourages exploration.
    value_coef:
        Coefficient on the value-function loss term.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "PPO"
    version = "v2"

    def __init__(
        self,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 8,
        update_freq: int = 512,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        np_seed = seed if seed is not None else 42
        self._np_rng = np.random.default_rng(np_seed)

        self._net = _ActorCritic(_N_STATE, hidden_size, self._np_rng)

        self._lr = lr
        self._gamma = gamma
        self._lam = lam
        self._clip_eps = clip_eps
        self._n_epochs = n_epochs
        self._update_freq = update_freq
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        # Running statistics for reward normalisation (exponential moving average).
        self._reward_ema_mean: float = 0.0
        self._reward_ema_var: float = 1.0
        self._reward_ema_alpha: float = 0.01

        # Rollout buffer (cleared after each PPO update).
        self._buf_states: list[np.ndarray] = []
        self._buf_actions: list[int] = []
        self._buf_log_probs: list[float] = []
        self._buf_rewards: list[float] = []
        self._buf_values: list[float] = []
        self._buf_dones: list[float] = []

        # State carried forward from the previous call.
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_log_prob: Optional[float] = None
        self._prev_value: Optional[float] = None
        self._prev_board: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction sampled from the PPO policy.

        On each call the agent:

        1. Records the reward from the previous step and appends the
           transition to the rollout buffer.
        2. Triggers a PPO update when ``update_freq`` steps have been
           collected.
        3. Runs the current policy to select a (possibly stochastic) action.
        """
        state = _encode_board(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

        # ----------------------------------------------------------------
        # Store transition
        # ----------------------------------------------------------------
        if self._prev_state is not None:
            raw_reward = _board_heuristic(board) - _board_heuristic(self._prev_board)  # type: ignore[arg-type]
            reward = self._normalize_reward(raw_reward)
            done = float(len(valid_dirs) == 0)
            self._buf_states.append(self._prev_state)
            self._buf_actions.append(self._prev_action)          # type: ignore[arg-type]
            self._buf_log_probs.append(self._prev_log_prob)      # type: ignore[arg-type]
            self._buf_rewards.append(float(reward))
            self._buf_values.append(self._prev_value)            # type: ignore[arg-type]
            self._buf_dones.append(done)

            if len(self._buf_states) >= self._update_freq:
                # Bootstrap the value of the current state.
                _, next_val, *_ = self._net.forward(state)
                self._ppo_update(float(next_val))
                self._buf_states.clear()
                self._buf_actions.clear()
                self._buf_log_probs.clear()
                self._buf_rewards.clear()
                self._buf_values.clear()
                self._buf_dones.clear()

        # ----------------------------------------------------------------
        # Policy forward pass — mask invalid actions
        # ----------------------------------------------------------------
        logits, value, *_ = self._net.forward(state)

        valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
        mask = np.full(_N_ACTIONS, -1e9, dtype=np.float32)
        mask[valid_idx] = 0.0
        probs = _softmax(logits + mask)

        action = int(self._np_rng.choice(_N_ACTIONS, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))

        # ----------------------------------------------------------------
        # Carry state forward
        # ----------------------------------------------------------------
        self._prev_state = state
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_value = float(value)
        self._prev_board = [row[:] for row in board]

        return DIRECTIONS[action]

    # ------------------------------------------------------------------
    # Reward normalisation
    # ------------------------------------------------------------------

    def _normalize_reward(self, reward: float) -> float:
        """Normalise *reward* using an exponential moving-average mean and variance.

        Stable gradient updates require rewards on a consistent scale.  This
        helper maintains a running EMA estimate of the reward distribution and
        returns an approximately zero-mean, unit-variance version of the input
        reward.  The normalisation improves as the EMA estimates converge over
        the course of training.
        """
        a = self._reward_ema_alpha
        self._reward_ema_mean = (1.0 - a) * self._reward_ema_mean + a * reward
        self._reward_ema_var = (
            (1.0 - a) * self._reward_ema_var
            + a * (reward - self._reward_ema_mean) ** 2
        )
        return (reward - self._reward_ema_mean) / (math.sqrt(self._reward_ema_var) + 1e-8)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, next_value: float) -> None:
        """Run ``n_epochs`` PPO-Clip updates over the collected rollout."""
        T = len(self._buf_states)
        if T == 0:
            return

        states = np.array(self._buf_states, dtype=np.float32)          # (T, 16)
        actions = np.array(self._buf_actions, dtype=np.int32)           # (T,)
        old_log_probs = np.array(self._buf_log_probs, dtype=np.float32) # (T,)
        rewards = np.array(self._buf_rewards, dtype=np.float32)         # (T,)
        values = np.array(self._buf_values, dtype=np.float32)           # (T,)
        dones = np.array(self._buf_dones, dtype=np.float32)             # (T,)

        # ------------------------------------------------------------------
        # Generalized Advantage Estimation (GAE)
        # ------------------------------------------------------------------
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self._gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self._gamma * self._lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Normalise advantages for stable updates.
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        # ------------------------------------------------------------------
        # PPO-Clip epochs
        # ------------------------------------------------------------------
        for _ in range(self._n_epochs):
            logits, value_pred, h2, z2, h1, z1 = self._net.forward(states)

            # Softmax probabilities and new log-probabilities.
            probs_all = _softmax(logits)                                   # (T, 4)
            log_probs_all = np.log(probs_all + 1e-8)                      # (T, 4)
            new_log_probs = log_probs_all[np.arange(T), actions]          # (T,)

            # PPO ratio and clipped surrogate objective.
            ratio = np.exp(new_log_probs - old_log_probs)                  # (T,)
            surr1 = ratio * advantages                                     # (T,)
            surr2 = np.clip(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * advantages
            policy_loss = -np.minimum(surr1, surr2).mean()

            # Value loss (MSE).
            value_loss = ((value_pred - returns) ** 2).mean()

            # Entropy bonus (encourages exploration).
            entropy = -(probs_all * log_probs_all).sum(axis=1).mean()

            # ------------------------------------------------------------------
            # Backpropagation
            # ------------------------------------------------------------------

            # --- Gradient of value loss ---
            # d(value_coef * value_loss) / d(value_pred[t]) = 2 * value_coef * (vp[t]-ret[t]) / T
            dL_dv = (2.0 * self._value_coef / T) * (value_pred - returns)  # (T,)

            # --- Gradient of policy loss w.r.t. new_log_probs ---
            # For min(surr1, surr2): gradient is -advantage / T when surr1 ≤ surr2 (unclipped),
            # 0 when clipped.
            clipped = (surr1 > surr2).astype(np.float32)  # 1 = clipped, 0 = unclipped
            dL_d_new_lp = -(1.0 - clipped) * advantages / T  # (T,)

            # --- Gradient of policy loss w.r.t. logits ---
            # d(log(softmax(l)[a])) / d(l) = e_a - softmax(l)
            dL_d_logits = np.zeros((T, _N_ACTIONS), dtype=np.float32)
            for i in range(T):
                grad = -probs_all[i].copy()
                grad[actions[i]] += 1.0
                dL_d_logits[i] = dL_d_new_lp[i] * grad

            # --- Gradient of entropy bonus w.r.t. logits ---
            # d(H) / d(logit_i) = p_i * (log(p_i) + H - log(p_i) - 1) ... simplifies to:
            # d(-sum_j p_j log p_j) / d(logit_i) = p_i * (sum_j p_j log p_j + log p_i)
            # Wait, proper derivation via Jacobian of softmax:
            # dH/d(logit_i) = p_i * (mean_entropy_t - (log_p_i + 1))  ... per sample:
            H_t = -(probs_all * log_probs_all).sum(axis=1, keepdims=True)  # (T,1)
            # dH_t/d(logit_i) = p_i * (-log_p_i - 1 + H_t + 1) = p_i * (H_t - log_p_i)
            # (derivation: apply Jacobian of softmax to -(log_p+1))
            dH_d_logits = probs_all * (H_t - log_probs_all)  # (T, 4)
            # d(-entropy_coef * mean_entropy) / d(logits) = -entropy_coef/T * dH_d_logits
            dL_d_logits += (-self._entropy_coef / T) * dH_d_logits

            # --- Actor head backprop ---
            dW_a = h2.T @ dL_d_logits                       # (hidden, 4)
            db_a = dL_d_logits.sum(axis=0)                  # (4,)
            dL_dh2 = dL_d_logits @ self._net.W_a.T          # (T, hidden)

            # --- Critic head backprop ---
            dW_v = h2.T @ dL_dv[:, None]                    # (hidden, 1)
            db_v = dL_dv.sum(keepdims=True)                 # (1,)
            dL_dh2 += dL_dv[:, None] @ self._net.W_v.T     # (T, hidden)

            # --- Shared hidden layer 2 backprop ---
            dz2 = dL_dh2 * _relu_grad(z2)                  # (T, hidden)
            dW2 = h1.T @ dz2                                # (hidden, hidden)
            db2 = dz2.sum(axis=0)                           # (hidden,)

            # --- Shared hidden layer 1 backprop ---
            dh1 = dz2 @ self._net.W2.T                      # (T, hidden)
            dz1 = dh1 * _relu_grad(z1)                      # (T, hidden)
            dW1 = states.T @ dz1                            # (16, hidden)
            db1 = dz1.sum(axis=0)                           # (hidden,)

            self._net.sgd_update(dW1, db1, dW2, db2, dW_a, db_a, dW_v, db_v, self._lr)
