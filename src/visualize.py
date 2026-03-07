"""Generate data visualizations from 2048 game results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def compute_run_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute run-level summary statistics from game results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.

    Returns
    -------
    dict
        Dictionary with keys:
        ``avg_score``, ``median_score``, ``p90_score``, ``max_score``,
        ``avg_moves``, ``avg_duration``, ``avg_best_tile``,
        ``win_rate``, ``games_per_second``.
    """
    total_duration = df["duration"].sum()
    n = len(df)
    return {
        "avg_score":      float(df["score"].mean()),
        "median_score":   float(df["score"].median()),
        "p90_score":      float(df["score"].quantile(0.9)),
        "max_score":      int(df["score"].max()),
        "avg_moves":      float(df["moves"].mean()),
        "avg_duration":   float(df["duration"].mean()),
        "avg_best_tile":  float(df["best_tile"].mean()),
        "win_rate":       float(df["won"].mean() * 100),
        "games_per_second": float(n / total_duration) if total_duration > 0 else 0.0,
    }


def score_distribution(df: pd.DataFrame, bins: int = 20) -> dict[str, int]:
    """Return a binned score distribution as ``{bin_label: count}``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.
    bins : int
        Number of histogram bins.

    Returns
    -------
    dict[str, int]
        Keys are bin range strings (e.g. ``"1000-2000"``), values are game counts.
    """
    counts, edges = np.histogram(df["score"], bins=bins)
    return {
        f"{int(edges[i])}-{int(edges[i + 1])}": int(counts[i])
        for i in range(len(counts))
    }


def moves_distribution(df: pd.DataFrame, bins: int = 20) -> dict[str, int]:
    """Return a binned move-count distribution as ``{bin_label: count}``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.
    bins : int
        Number of histogram bins.

    Returns
    -------
    dict[str, int]
        Keys are bin range strings (e.g. ``"100-200"``), values are game counts.
    """
    counts, edges = np.histogram(df["moves"], bins=bins)
    return {
        f"{int(edges[i])}-{int(edges[i + 1])}": int(counts[i])
        for i in range(len(counts))
    }


def tile_distribution(df: pd.DataFrame) -> dict[int, int]:
    """Return the frequency of each best-tile value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.

    Returns
    -------
    dict[int, int]
        Maps tile value → number of games ending with that tile as the best tile.
        E.g. ``{64: 120, 128: 300, 256: 180}``.
    """
    return {int(k): int(v) for k, v in df["best_tile"].value_counts().sort_index().items()}


def plot_results(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    output_stem: str | None = None,
) -> None:
    """
    Save a set of charts summarizing game results to *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.
    output_dir : str or Path
        Directory where PNG files will be written (created if needed).
    output_stem : str, optional
        Filename stem (without extension) for the saved chart.
        Defaults to ``results_<algorithm_lowercase>`` when *None*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo = df["algorithm"].iloc[0] if not df.empty else "Unknown"
    n = len(df)
    if output_stem is None:
        output_stem = f"results_{algo.lower()}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"2048 Results — {algo} Algorithm  (n={n})", fontsize=15, fontweight="bold")

    # 1. Score distribution
    ax = axes[0, 0]
    ax.hist(df["score"], bins=20, color="#f59563", edgecolor="white")
    ax.axvline(df["score"].mean(), color="#776e65", linestyle="--", label=f"mean = {df['score'].mean():,.0f}")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Games")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()

    # 2. Max tile frequency
    ax = axes[0, 1]
    tile_counts = df["best_tile"].value_counts().sort_index()
    bars = ax.bar([str(t) for t in tile_counts.index], tile_counts.values, color="#edcf72", edgecolor="white")
    ax.set_title("Highest Tile Achieved")
    ax.set_xlabel("Tile Value")
    ax.set_ylabel("Games")
    for bar, val in zip(bars, tile_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=9)

    # 3. Move count distribution
    ax = axes[1, 0]
    ax.hist(df["moves"], bins=20, color="#8f7a66", edgecolor="white")
    ax.axvline(df["moves"].mean(), color="#f9f6f2", linestyle="--",
               label=f"mean = {df['moves'].mean():,.0f}")
    ax.set_title("Move Count Distribution")
    ax.set_xlabel("Moves")
    ax.set_ylabel("Games")
    ax.legend()

    # 4. Score over games (rolling mean)
    ax = axes[1, 1]
    ax.plot(df["game_index"], df["score"], alpha=0.4, color="#f59563", label="score")
    window = max(1, n // 5)
    rolling = df["score"].rolling(window, min_periods=1).mean()
    ax.plot(df["game_index"], rolling, color="#776e65", linewidth=2,
            label=f"rolling mean (w={window})")
    ax.set_title("Score per Game")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / f"{output_stem}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved → {out_path}")

    # Also print a text summary
    _print_summary(df, algo)


def _print_summary(df: pd.DataFrame, algo: str) -> None:
    """Print a concise text summary to stdout."""
    metrics = compute_run_metrics(df)
    print("\n" + "=" * 60)
    print(f"  Summary — {algo} Algorithm  (n={len(df)} games)")
    print("=" * 60)
    print(f"  Score   :  avg={metrics['avg_score']:>8,.0f}  median={metrics['median_score']:>8,.0f}"
          f"  p90={metrics['p90_score']:>8,.0f}  max={metrics['max_score']:>6,}")
    print(f"  Best tile:  avg={metrics['avg_best_tile']:>6.1f}  max={df['best_tile'].max():>6}")
    print(f"  Moves   :  avg={metrics['avg_moves']:>8.1f}  max={df['moves'].max():>6}")
    print(f"  Duration:  avg={metrics['avg_duration']:>6.3f}s")
    print(f"  Win rate:  {metrics['win_rate']:.1f}%")
    print(f"  Throughput: {metrics['games_per_second']:.2f} games/sec")
    print("=" * 60 + "\n")
