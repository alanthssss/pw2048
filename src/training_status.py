"""Training status helpers — inspect checkpoints and training logs.

Two entry points are exposed:

* :func:`inspect_checkpoint` — print a human-readable summary of the scalar
  state and network weight statistics stored in a ``.npz`` checkpoint file.

* :func:`print_training_status` — read a ``training_log.csv`` produced by
  :class:`~src.rl_trainer.TrainingLogger` and print a convergence summary
  with recent score trends and an overall stability assessment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint inspector
# ---------------------------------------------------------------------------

def inspect_checkpoint(path: str | Path, *, verbose: bool = True) -> dict:
    """Load a ``.npz`` checkpoint and return (and optionally print) its stats.

    Works for both DQN-v3 and PPO-v3 checkpoints (auto-detected from the
    array keys present in the file).

    Parameters
    ----------
    path:
        Path to the ``.npz`` checkpoint file.
    verbose:
        When ``True`` (default), print a formatted summary to stdout.

    Returns
    -------
    dict
        Keys: ``algo``, ``step``, ``epsilon`` (DQN only), ``file_size_kb``,
        ``n_params``, ``weight_norms``, ``adam_t``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    d = np.load(path)
    keys = set(d.files)
    file_size_kb = path.stat().st_size / 1024

    # ── Detect algorithm ────────────────────────────────────────────────────
    is_dqn = "q_W1" in keys          # DQN has q_* (online Q-net) arrays
    is_ppo = "a_W1" in keys or "pi_W1" in keys   # PPO actor arrays
    # Older PPO checkpoints may use "actor_W1" or just "W1"
    if not is_dqn and not is_ppo:
        is_ppo = any(k.startswith("a_") or "actor" in k or "critic" in k for k in keys)
    algo = "DQN-v3" if is_dqn else ("PPO-v3" if is_ppo else "unknown")

    # ── Scalar training state ────────────────────────────────────────────────
    step    = int(d["step"][0])    if "step"    in keys else None
    epsilon = float(d["epsilon"][0]) if "epsilon" in keys else None
    adam_t  = int(d["adam_t"][0])  if "adam_t"  in keys else None

    # ── Weight statistics ────────────────────────────────────────────────────
    weight_arrays = {k: d[k] for k in keys if k[0] in ("q", "t", "a", "W", "b", "p", "c")
                     and k not in ("step", "epsilon", "adam_t")}
    # Filter to genuine weight arrays (must be 2-D matrices or 1-D biases)
    weight_arrays = {k: v for k, v in weight_arrays.items()
                     if v.ndim in (1, 2) and v.size > 1}

    n_params = sum(v.size for v in weight_arrays.values())
    weight_norms = {k: float(np.linalg.norm(v)) for k, v in sorted(weight_arrays.items())}

    # Compute per-layer L2 norm statistics
    norms = list(weight_norms.values())
    norm_mean  = float(np.mean(norms))  if norms else 0.0
    norm_min   = float(np.min(norms))   if norms else 0.0
    norm_max   = float(np.max(norms))   if norms else 0.0

    result = {
        "algo":        algo,
        "step":        step,
        "epsilon":     epsilon,
        "file_size_kb": round(file_size_kb, 1),
        "n_params":    n_params,
        "weight_norms": weight_norms,
        "adam_t":      adam_t,
    }

    if verbose:
        _print_checkpoint_summary(path, result, norm_mean, norm_min, norm_max)

    return result


def _print_checkpoint_summary(
    path: Path,
    r: dict,
    norm_mean: float,
    norm_min: float,
    norm_max: float,
) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  Checkpoint: {path}")
    print(sep)
    print(f"  Algorithm   : {r['algo']}")
    print(f"  File size   : {r['file_size_kb']:.1f} KB")
    print(f"  Parameters  : {r['n_params']:,}")
    if r["step"] is not None:
        print(f"  Global step : {r['step']:,}")
    if r["epsilon"] is not None:
        pct = r["epsilon"] * 100
        # Interpret epsilon as exploration stage
        if pct > 20:
            stage = "early exploration"
        elif pct > 5:
            stage = "mid training"
        else:
            stage = "late / converging"
        print(f"  ε (epsilon) : {pct:.2f}%  [{stage}]")
    if r["adam_t"] is not None:
        print(f"  Adam steps  : {r['adam_t']:,}")
    print(f"  Weight norms: mean={norm_mean:.4f}  min={norm_min:.4f}  max={norm_max:.4f}")
    print(sep)
    # Print per-array norms in columns
    items = sorted(r["weight_norms"].items())
    for i in range(0, len(items), 3):
        row = items[i:i+3]
        print("  " + "   ".join(f"{k:<12} {v:>8.4f}" for k, v in row))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Training log status
# ---------------------------------------------------------------------------

def print_training_status(
    log_dir: str | Path,
    *,
    window: int = 10,
    verbose: bool = True,
) -> dict:
    """Read ``training_log.csv`` and print a convergence / status summary.

    Parameters
    ----------
    log_dir:
        Directory that contains ``training_log.csv`` (the value passed to
        ``--tensorboard-dir`` during training).  Can also be the direct path
        to a ``training_log.csv`` file.
    window:
        Number of most-recent *eval* rounds to analyse for the trend.
        Defaults to 10.
    verbose:
        When ``True`` (default), print a formatted summary.

    Returns
    -------
    dict
        Keys: ``total_train_games``, ``best_eval_score``, ``recent_eval_mean``,
        ``trend``, ``stable``, ``eval_rounds``.
    """
    log_dir = Path(log_dir)
    csv_path = log_dir if log_dir.suffix == ".csv" else log_dir / "training_log.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"training_log.csv not found at {csv_path}.\n"
            "Make sure you passed --tensorboard-dir during training."
        )

    # ── Parse CSV ────────────────────────────────────────────────────────────
    data: dict[str, list] = {}
    import csv as _csv
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            # Skip malformed rows (missing columns or non-integer step).
            try:
                tag  = row["tag"]
                step = int(float(row["step"]))   # accept both "5" and "5.0"
                val  = float(row["value"])
            except (KeyError, ValueError, TypeError):
                continue
            if tag not in data:
                data[tag] = []
            data[tag].append((step, val))

    # ── Train scores ─────────────────────────────────────────────────────────
    train_scores = [v for _, v in data.get("train/score", [])]
    total_train_games = len(train_scores)
    recent_train = train_scores[-50:] if train_scores else []

    # ── Eval scores ──────────────────────────────────────────────────────────
    eval_means = [v for _, v in data.get("eval/mean_score", [])]
    eval_rounds = len(eval_means)
    best_eval   = max(eval_means) if eval_means else float("nan")
    recent_eval = eval_means[-window:] if len(eval_means) >= window else eval_means
    recent_eval_mean = float(np.mean(recent_eval)) if recent_eval else float("nan")

    # ── Trend: linear regression slope over recent eval rounds ───────────────
    trend = float("nan")
    stable = None
    if len(recent_eval) >= 2:
        x = np.arange(len(recent_eval), dtype=float)
        slope, _ = np.polyfit(x, recent_eval, 1)
        trend = float(slope)
        # "Stable" = slope is small relative to recent mean
        rel_slope = abs(trend) / (abs(recent_eval_mean) + 1e-8)
        stable = rel_slope < 0.005  # less than 0.5% change per eval round

    # ── Epsilon (exploration rate) if logged ─────────────────────────────────
    eps_series = [v for _, v in data.get("train/epsilon", [])]
    current_eps: Optional[float] = eps_series[-1] if eps_series else None

    result = {
        "total_train_games": total_train_games,
        "best_eval_score":   best_eval,
        "recent_eval_mean":  recent_eval_mean,
        "trend":             trend,
        "stable":            stable,
        "eval_rounds":       eval_rounds,
        "current_epsilon":   current_eps,
        "recent_train_mean": float(np.mean(recent_train)) if recent_train else float("nan"),
    }

    if verbose:
        _print_status_summary(csv_path, result, window, recent_eval, eval_means)

    return result


def _print_status_summary(
    csv_path: Path,
    r: dict,
    window: int,
    recent_eval: list,
    all_eval: list,
) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  Training status: {csv_path.parent}")
    print(sep)
    print(f"  Train games    : {r['total_train_games']:,}")
    print(f"  Eval rounds    : {r['eval_rounds']}")

    if not math.isnan(r["best_eval_score"]):
        print(f"  Best eval score: {r['best_eval_score']:.0f}")
    if not math.isnan(r["recent_eval_mean"]):
        print(f"  Recent eval mean ({min(window, r['eval_rounds'])} rounds): {r['recent_eval_mean']:.0f}")
    if not math.isnan(r["recent_train_mean"]):
        print(f"  Recent train mean (last 50 games): {r['recent_train_mean']:.0f}")

    if r["current_epsilon"] is not None:
        print(f"  Current ε      : {r['current_epsilon'] * 100:.2f}%")

    if not math.isnan(r["trend"]):
        direction = "↑ improving" if r["trend"] > 0 else "↓ declining"
        print(f"  Score trend    : {r['trend']:+.1f} pts/eval  [{direction}]")
        if r["stable"] is not None:
            status = "🟢 STABLE  (plateau detected — consider early stopping)"  \
                     if r["stable"] else                                         \
                     "🟡 IMPROVING  (still learning — keep training)"
            print(f"  Convergence    : {status}")

    # Mini sparkline of recent eval scores
    if len(recent_eval) >= 2:
        hi = max(recent_eval)
        lo = min(recent_eval)
        rng = hi - lo or 1.0
        bars = " ▁▂▃▄▅▆▇█"
        spark = "".join(bars[min(8, int(8 * (v - lo) / rng))] for v in recent_eval)
        print(f"  Score sparkline: {spark}")

    # Percentile summary of all eval rounds
    if len(all_eval) >= 4:
        p25, p50, p75 = np.percentile(all_eval, [25, 50, 75])
        print(f"  Score P25/50/75: {p25:.0f} / {p50:.0f} / {p75:.0f}")

    print(sep + "\n")
