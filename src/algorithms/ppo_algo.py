"""PPO algorithm: Proximal Policy Optimization reinforcement learning for 2048."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Union

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
# PPO algorithm – version 1 (original PPO-Clip)
# ---------------------------------------------------------------------------

class PPOAlgorithmV1(BaseAlgorithm):
    """Proximal Policy Optimization – version 1 (original PPO-Clip).

    Collects a fixed-length rollout and performs ``n_epochs`` passes of
    PPO-Clip updates.  Raw heuristic-delta rewards are used without any
    normalisation; see :class:`PPOAlgorithmV2` for the EMA-normalised variant.

    Parameters
    ----------
    hidden_size:
        Number of units in each shared hidden layer (default 128).
    lr:
        Learning rate for SGD updates.
    gamma:
        Discount factor used in GAE advantage estimation.
    lam:
        λ parameter for Generalized Advantage Estimation (GAE).
    clip_eps:
        PPO clipping parameter ε.
    n_epochs:
        Number of gradient-update epochs per rollout (default 4).
    update_freq:
        Number of steps per rollout (default 256).
    entropy_coef:
        Coefficient on the entropy bonus.
    value_coef:
        Coefficient on the value-function loss.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "PPO-v1"
    version = "v1"

    def __init__(
        self,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        update_freq: int = 256,
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

        self._buf_states: list[np.ndarray] = []
        self._buf_actions: list[int] = []
        self._buf_log_probs: list[float] = []
        self._buf_rewards: list[float] = []
        self._buf_values: list[float] = []
        self._buf_dones: list[float] = []

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_log_prob: Optional[float] = None
        self._prev_value: Optional[float] = None
        self._prev_board: Optional[List[List[int]]] = None

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction sampled from the PPO-v1 policy."""
        state = _encode_board(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

        if self._prev_state is not None:
            # PPO-v1 uses the raw heuristic-delta reward with no further
            # normalisation — the key difference from PPO-v2's EMA approach.
            reward = _board_heuristic(board) - _board_heuristic(self._prev_board)  # type: ignore[arg-type]
            done = float(len(valid_dirs) == 0)
            self._buf_states.append(self._prev_state)
            self._buf_actions.append(self._prev_action)          # type: ignore[arg-type]
            self._buf_log_probs.append(self._prev_log_prob)      # type: ignore[arg-type]
            self._buf_rewards.append(float(reward))
            self._buf_values.append(self._prev_value)            # type: ignore[arg-type]
            self._buf_dones.append(done)

            if len(self._buf_states) >= self._update_freq:
                _, next_val, *_ = self._net.forward(state)
                self._ppo_update(float(next_val))
                self._buf_states.clear()
                self._buf_actions.clear()
                self._buf_log_probs.clear()
                self._buf_rewards.clear()
                self._buf_values.clear()
                self._buf_dones.clear()

        logits, value, *_ = self._net.forward(state)
        valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
        mask = np.full(_N_ACTIONS, -1e9, dtype=np.float32)
        mask[valid_idx] = 0.0
        probs = _softmax(logits + mask)

        action = int(self._np_rng.choice(_N_ACTIONS, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))

        self._prev_state = state
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_value = float(value)
        self._prev_board = [row[:] for row in board]

        return DIRECTIONS[action]

    def _ppo_update(self, next_value: float) -> None:
        """Run ``n_epochs`` PPO-Clip updates over the collected rollout."""
        T = len(self._buf_states)
        if T == 0:
            return

        states = np.array(self._buf_states, dtype=np.float32)
        actions = np.array(self._buf_actions, dtype=np.int32)
        old_log_probs = np.array(self._buf_log_probs, dtype=np.float32)
        rewards = np.array(self._buf_rewards, dtype=np.float32)
        values = np.array(self._buf_values, dtype=np.float32)
        dones = np.array(self._buf_dones, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self._gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self._gamma * self._lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        for _ in range(self._n_epochs):
            logits_b, values_b, h2_b, z2_b, h1_b, z1_b = self._net.forward(states)
            probs_b = _softmax(logits_b)
            log_probs_b = np.log(probs_b[np.arange(T), actions] + 1e-8)
            entropy_b = -(probs_b * np.log(probs_b + 1e-8)).sum(axis=-1)
            ratios = np.exp(log_probs_b - old_log_probs)
            clip_ratios = np.clip(ratios, 1.0 - self._clip_eps, 1.0 + self._clip_eps)
            policy_loss = -np.minimum(ratios * advantages, clip_ratios * advantages).mean()
            value_loss = ((values_b - returns) ** 2).mean()
            entropy_loss = -entropy_b.mean()
            total_loss = policy_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss

            dL_dlogits = probs_b.copy()
            dL_dlogits[np.arange(T), actions] -= 1.0
            pg_weights = np.where(
                (ratios > 1.0 + self._clip_eps) | (ratios < 1.0 - self._clip_eps),
                0.0, -advantages,
            )
            dL_dlogits = dL_dlogits * pg_weights[:, None] / T
            dL_dlogits -= self._entropy_coef * (
                np.eye(_N_ACTIONS)[actions] - probs_b
            ) / T

            dL_dvalue = 2.0 * self._value_coef * (values_b - returns) / T
            dL_dh2 = dL_dlogits @ self._net.W_a.T + (dL_dvalue[:, None] * self._net.W_v.T)
            dW_a = h2_b.T @ dL_dlogits
            db_a = dL_dlogits.sum(axis=0)
            dW_v = (h2_b * dL_dvalue[:, None]).sum(axis=0, keepdims=True).T
            db_v = np.array([dL_dvalue.sum()])
            dz2 = dL_dh2 * _relu_grad(z2_b)
            dW2 = h1_b.T @ dz2
            db2 = dz2.sum(axis=0)
            dh1 = dz2 @ self._net.W2.T
            dz1 = dh1 * _relu_grad(z1_b)
            dW1 = states.T @ dz1
            db1 = dz1.sum(axis=0)
            self._net.sgd_update(dW1, db1, dW2, db2, dW_a, db_a, dW_v, db_v, self._lr)


# ---------------------------------------------------------------------------
# PPO algorithm – version 2 (PPO-Clip with EMA reward normalisation)
# ---------------------------------------------------------------------------

class PPOAlgorithmV2(BaseAlgorithm):
    """Proximal Policy Optimization – version 2.

    Improves on :class:`PPOAlgorithmV1` by normalising rewards with an
    exponential moving-average estimate, using a wider hidden layer (256) and
    more update epochs (8) for better sample efficiency.

    Parameters
    ----------
    hidden_size:
        Number of units in each shared hidden layer (default 256).
    lr:
        Learning rate for SGD updates.
    gamma:
        Discount factor used in GAE advantage estimation.
    lam:
        λ parameter for Generalized Advantage Estimation (GAE).
    clip_eps:
        PPO clipping parameter ε.
    n_epochs:
        Number of gradient-update epochs per rollout (default 8).
    update_freq:
        Number of steps per rollout (default 512).
    entropy_coef:
        Coefficient on the entropy bonus.
    value_coef:
        Coefficient on the value-function loss.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "PPO-v2"
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


# ---------------------------------------------------------------------------
# Backward-compatible alias — PPOAlgorithm always points to the latest version.
# ---------------------------------------------------------------------------

PPOAlgorithm = PPOAlgorithmV2


# ---------------------------------------------------------------------------
# V3 helpers: one-hot encoding, score-based reward, Adam optimizer
# (defined locally so ppo_algo.py stays self-contained)
# ---------------------------------------------------------------------------

_N_LEVELS = 16
_N_STATE_V3 = 16 * _N_LEVELS  # 256


def _encode_board_onehot(board: List[List[int]]) -> np.ndarray:
    """One-hot encode the board state as a 256-dim float32 vector.

    Each of the 16 cells is represented by a 16-level one-hot vector.
    The 16 vectors are concatenated into a 256-dim output.
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

    Replaces the broken V1/V2 heuristic-delta reward.  Components:

    * ``log₂(merge_score + 1)``: 0 for non-merging moves; grows with tile
      value (e.g. ~4 for two 8s→16, ~8 for two 128s→256).
    * ``0.1 · empty_count``: small bonus for keeping the board open.
    """
    _, score = simulate_move(prev_board, DIRECTIONS[action_idx])
    empty = sum(1 for r in range(4) for c in range(4) if curr_board[r][c] == 0)
    return math.log2(score + 1) + 0.1 * empty


class _Adam:
    """Adam optimizer for pure-NumPy parameter arrays."""

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
        """Apply one Adam step; each entry is ``(name, param, grad)``."""
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
# PPO algorithm – version 3 (PPO-Clip + Adam + one-hot state + score reward)
# ---------------------------------------------------------------------------

class PPOAlgorithmV3(BaseAlgorithm):
    """Proximal Policy Optimization – version 3.

    This version fixes all major shortcomings of V1/V2:

    * **One-hot state encoding** (256-dim) instead of 16-dim normalized log₂.
    * **Score-based reward**: ``log₂(merge_score+1) + 0.1·empty_tiles``
      instead of the V1/V2 heuristic-delta which *penalised* high-value
      merges and caused the agent to score below Random.
    * **Adam optimizer**: replaces vanilla SGD.
    * **Fixed ``done`` flag**: computed before the invalid-move fallback so
      terminal states are correctly signalled.
    * **Game-boundary reset** via :meth:`on_game_start`: prevents corrupt
      cross-game transitions from polluting the rollout buffer.
    * **Behavioural-cloning pre-training**: before any RL experience the
      actor head is warmed up by imitating the Heuristic algorithm on
      ``n_pretrain_games`` simulated games (browser-free, in-process).

    Parameters
    ----------
    hidden_size:
        Number of units in each shared hidden layer.
    lr:
        Adam learning rate.
    gamma:
        Discount factor for GAE.
    lam:
        λ for Generalized Advantage Estimation.
    clip_eps:
        PPO clipping parameter ε.
    n_epochs:
        Gradient-update epochs per rollout.
    update_freq:
        Number of steps per rollout.
    entropy_coef:
        Entropy bonus coefficient (higher → more exploration).
    value_coef:
        Value-loss coefficient.
    n_pretrain_games:
        Number of in-process heuristic games used for behavioural-cloning
        pre-training at construction time.  Set to 0 to disable.
    seed:
        Optional RNG seed.
    """

    name = "PPO-v3"
    version = "v3"
    #: Indicates this algorithm supports save/load checkpoint.
    supports_checkpoint: bool = True

    def __init__(
        self,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        update_freq: int = 256,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        n_pretrain_games: int = 50,
        seed: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._rng = random.Random(seed)
        np_seed = seed if seed is not None else 42
        self._np_rng = np.random.default_rng(np_seed)

        self._net = _ActorCritic(_N_STATE_V3, hidden_size, self._np_rng)
        self._optimizer = _Adam(lr=lr)

        self._gamma = gamma
        self._lam = lam
        self._clip_eps = clip_eps
        self._n_epochs = n_epochs
        self._update_freq = update_freq
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        self._buf_states: list[np.ndarray] = []
        self._buf_actions: list[int] = []
        self._buf_log_probs: list[float] = []
        self._buf_rewards: list[float] = []
        self._buf_values: list[float] = []
        self._buf_dones: list[float] = []

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_log_prob: Optional[float] = None
        self._prev_value: Optional[float] = None
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
        """Flush the rollout buffer and reset per-game transition state."""
        self._buf_states.clear()
        self._buf_actions.clear()
        self._buf_log_probs.clear()
        self._buf_rewards.clear()
        self._buf_values.clear()
        self._buf_dones.clear()
        self._prev_state = None
        self._prev_action = None
        self._prev_log_prob = None
        self._prev_value = None
        self._prev_board = None

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the next direction sampled from the PPO-v3 policy."""
        state = _encode_board_onehot(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        # Compute done flag *before* the fallback override.
        is_terminal = len(valid_dirs) == 0
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)

        # ----------------------------------------------------------------
        # Store transition and trigger PPO update when buffer is full
        # ----------------------------------------------------------------
        if self._prev_state is not None:
            reward = _score_reward(self._prev_board, self._prev_action, board)  # type: ignore[arg-type]
            done = float(is_terminal)
            self._buf_states.append(self._prev_state)
            self._buf_actions.append(self._prev_action)          # type: ignore[arg-type]
            self._buf_log_probs.append(self._prev_log_prob)      # type: ignore[arg-type]
            self._buf_rewards.append(float(reward))
            self._buf_values.append(self._prev_value)            # type: ignore[arg-type]
            self._buf_dones.append(done)

            if len(self._buf_states) >= self._update_freq:
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

        self._prev_state = state
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_value = float(value)
        self._prev_board = [row[:] for row in board]

        return DIRECTIONS[action]

    def predict(self, board: List[List[int]]) -> str:
        """Return the greedy (argmax) action without any buffer update.

        Used by :class:`src.rl_trainer.EvalCallback` for deterministic
        evaluation episodes.  Never writes to the rollout buffer, does not
        update the previous-transition cache, and does not trigger a PPO
        update.

        Parameters
        ----------
        board : list[list[int]]
            Current 4×4 board.

        Returns
        -------
        str
            The direction with the highest policy logit among valid moves
            (i.e. the greedy/deterministic policy action).
        """
        state = _encode_board_onehot(board)
        valid_dirs = [
            d for d in DIRECTIONS
            if not _boards_equal(board, simulate_move(board, d)[0])
        ]
        if not valid_dirs:
            valid_dirs = list(DIRECTIONS)
        logits, _, *_ = self._net.forward(state)
        valid_idx = [_DIR_INDEX[d] for d in valid_dirs]
        mask = np.full(_N_ACTIONS, -1e9, dtype=np.float32)
        mask[valid_idx] = 0.0
        best = int(np.argmax(logits + mask))
        return DIRECTIONS[best]

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, next_value: float) -> None:
        """Run ``n_epochs`` PPO-Clip updates over the collected rollout."""
        T = len(self._buf_states)
        if T == 0:
            return

        states = np.array(self._buf_states, dtype=np.float32)           # (T, 256)
        actions = np.array(self._buf_actions, dtype=np.int32)            # (T,)
        old_log_probs = np.array(self._buf_log_probs, dtype=np.float32)  # (T,)
        rewards = np.array(self._buf_rewards, dtype=np.float32)          # (T,)
        values = np.array(self._buf_values, dtype=np.float32)            # (T,)
        dones = np.array(self._buf_dones, dtype=np.float32)              # (T,)

        # Generalized Advantage Estimation (GAE)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self._gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self._gamma * self._lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self._n_epochs):
            logits, value_pred, h2, z2, h1, z1 = self._net.forward(states)
            probs_all = _softmax(logits)                                   # (T, 4)
            log_probs_all = np.log(probs_all + 1e-8)                      # (T, 4)
            new_log_probs = log_probs_all[np.arange(T), actions]          # (T,)

            ratio = np.exp(new_log_probs - old_log_probs)                  # (T,)
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * advantages

            # --- Gradient of value loss ---
            dL_dv = (2.0 * self._value_coef / T) * (value_pred - returns)  # (T,)

            # --- Gradient of policy loss w.r.t. new log-probs ---
            clipped = (surr1 > surr2).astype(np.float32)
            dL_d_new_lp = -(1.0 - clipped) * advantages / T               # (T,)

            # --- Gradient of policy loss w.r.t. logits ---
            dL_d_logits = np.zeros((T, _N_ACTIONS), dtype=np.float32)
            for i in range(T):
                grad = -probs_all[i].copy()
                grad[actions[i]] += 1.0
                dL_d_logits[i] = dL_d_new_lp[i] * grad

            # --- Entropy bonus gradient w.r.t. logits ---
            H_t = -(probs_all * log_probs_all).sum(axis=1, keepdims=True)  # (T, 1)
            dH_d_logits = probs_all * (H_t - log_probs_all)               # (T, 4)
            dL_d_logits += (-self._entropy_coef / T) * dH_d_logits

            # --- Actor head ---
            dW_a = h2.T @ dL_d_logits                                      # (hidden, 4)
            db_a = dL_d_logits.sum(axis=0)                                 # (4,)
            dL_dh2 = dL_d_logits @ self._net.W_a.T                        # (T, hidden)

            # --- Critic head ---
            dW_v = h2.T @ dL_dv[:, None]                                   # (hidden, 1)
            db_v = dL_dv.sum(keepdims=True)                                # (1,)
            dL_dh2 += dL_dv[:, None] @ self._net.W_v.T                    # (T, hidden)

            # --- Shared hidden layer 2 ---
            dz2 = dL_dh2 * _relu_grad(z2)                                  # (T, hidden)
            dW2 = h1.T @ dz2                                               # (hidden, hidden)
            db2 = dz2.sum(axis=0)                                          # (hidden,)

            # --- Shared hidden layer 1 ---
            dh1 = dz2 @ self._net.W2.T                                     # (T, hidden)
            dz1 = dh1 * _relu_grad(z1)                                     # (T, hidden)
            dW1 = states.T @ dz1                                           # (256, hidden)
            db1 = dz1.sum(axis=0)                                          # (hidden,)

            self._optimizer.step([
                ("W1",   self._net.W1,   dW1),
                ("b1",   self._net.b1,   db1),
                ("W2",   self._net.W2,   dW2),
                ("b2",   self._net.b2,   db2),
                ("W_a",  self._net.W_a,  dW_a),
                ("b_a",  self._net.b_a,  db_a),
                ("W_v",  self._net.W_v,  dW_v),
                ("b_v",  self._net.b_v,  db_v),
            ])

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist all trainable state to a ``.npz`` file.

        Saved state
        -----------
        * Actor-critic network weights
        * Adam optimizer state (step counter, first and second moment vectors)

        The rollout buffer is intentionally **not** saved — PPO is on-policy
        so stale rollout data from a previous run would be invalid.

        Parameters
        ----------
        path:
            Destination file.  A ``.npz`` extension is recommended.
        """
        data: dict[str, np.ndarray] = {
            # Shared hidden layers
            "W1":  self._net.W1,
            "b1":  self._net.b1,
            "W2":  self._net.W2,
            "b2":  self._net.b2,
            # Actor head
            "W_a": self._net.W_a,
            "b_a": self._net.b_a,
            # Critic head
            "W_v": self._net.W_v,
            "b_v": self._net.b_v,
            # Adam step counter
            "adam_t": np.array([self._optimizer._t], dtype=np.int64),
        }
        for k, v in self._optimizer._m.items():
            data[f"adam_m_{k}"] = v
        for k, v in self._optimizer._v.items():
            data[f"adam_v_{k}"] = v
        np.savez(path, **data)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore trainable state saved by :meth:`save_checkpoint`.

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
        self._net.W1  = d["W1"].copy()
        self._net.b1  = d["b1"].copy()
        self._net.W2  = d["W2"].copy()
        self._net.b2  = d["b2"].copy()
        self._net.W_a = d["W_a"].copy()
        self._net.b_a = d["b_a"].copy()
        self._net.W_v = d["W_v"].copy()
        self._net.b_v = d["b_v"].copy()
        self._optimizer._t = int(d["adam_t"][0])
        self._optimizer._m = {
            k[7:]: d[k].copy() for k in d.files if k.startswith("adam_m_")
        }
        self._optimizer._v = {
            k[7:]: d[k].copy() for k in d.files if k.startswith("adam_v_")
        }

    # ------------------------------------------------------------------
    # Behavioural-cloning pre-training
    # ------------------------------------------------------------------

    def _pretrain_bc(self, n_games: int) -> None:
        """Warm-start the actor by imitating the Heuristic algorithm.

        Runs ``n_games`` 2048 games in-process (no browser) using the
        Heuristic policy to pick moves and tile-spawn simulation to advance
        board state.  Collected ``(state, heuristic_action)`` pairs are used
        for supervised cross-entropy training of the actor head, which gives
        the agent a strong starting point before any RL experience.
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

        # Three supervised passes over the collected data.
        for _epoch in range(3):
            perm = self._np_rng.permutation(n)
            for start in range(0, n - bc_batch + 1, bc_batch):
                S = states[perm[start:start + bc_batch]]       # (B, 256)
                A = actions[perm[start:start + bc_batch]]      # (B,)
                B = len(S)

                logits, _, h2, z2, h1, z1 = self._net.forward(S)

                # Softmax cross-entropy on actor logits: dL/dl = (softmax(l) − one_hot(a)) / B
                l_shifted = logits - logits.max(axis=1, keepdims=True)
                exp_l = np.exp(l_shifted)
                probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)
                dL_d_logits = probs.copy()
                dL_d_logits[np.arange(B), A] -= 1.0
                dL_d_logits /= B

                # Actor head gradients only; critic head gets zero gradient.
                dW_a = h2.T @ dL_d_logits                      # (hidden, 4)
                db_a = dL_d_logits.sum(axis=0)                 # (4,)
                dL_dh2 = dL_d_logits @ self._net.W_a.T         # (B, hidden)

                dz2 = dL_dh2 * _relu_grad(z2)                  # (B, hidden)
                dW2 = h1.T @ dz2                               # (hidden, hidden)
                db2 = dz2.sum(axis=0)                          # (hidden,)
                dh1 = dz2 @ self._net.W2.T                     # (B, hidden)
                dz1 = dh1 * _relu_grad(z1)                     # (B, hidden)
                dW1 = S.T @ dz1                                # (256, hidden)
                db1 = dz1.sum(axis=0)                          # (hidden,)

                self._optimizer.step([
                    ("W1",   self._net.W1,   dW1),
                    ("b1",   self._net.b1,   db1),
                    ("W2",   self._net.W2,   dW2),
                    ("b2",   self._net.b2,   db2),
                    ("W_a",  self._net.W_a,  dW_a),
                    ("b_a",  self._net.b_a,  db_a),
                ])
