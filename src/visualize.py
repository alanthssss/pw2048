"""Generate data visualizations from 2048 game results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def plot_results(df: pd.DataFrame, output_dir: str | Path = "results") -> None:
    """
    Save a set of charts summarizing game results to *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`src.runner.run_games`.
    output_dir : str or Path
        Directory where PNG files will be written (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo = df["algorithm"].iloc[0] if not df.empty else "Unknown"
    n = len(df)

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
    tile_counts = df["max_tile"].value_counts().sort_index()
    bars = ax.bar([str(t) for t in tile_counts.index], tile_counts.values, color="#edcf72", edgecolor="white")
    ax.set_title("Highest Tile Achieved")
    ax.set_xlabel("Tile Value")
    ax.set_ylabel("Games")
    for bar, val in zip(bars, tile_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=9)

    # 3. Move count distribution
    ax = axes[1, 0]
    ax.hist(df["move_count"], bins=20, color="#8f7a66", edgecolor="white")
    ax.axvline(df["move_count"].mean(), color="#f9f6f2", linestyle="--",
               label=f"mean = {df['move_count'].mean():,.0f}")
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
    out_path = output_dir / f"results_{algo.lower()}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved → {out_path}")

    # Also print a text summary
    _print_summary(df, algo)


def _print_summary(df: pd.DataFrame, algo: str) -> None:
    """Print a concise text summary to stdout."""
    print("\n" + "=" * 50)
    print(f"  Summary — {algo} Algorithm  (n={len(df)} games)")
    print("=" * 50)
    print(f"  Score   :  min={df['score'].min():>6,}  mean={df['score'].mean():>8,.0f}  max={df['score'].max():>6,}")
    print(f"  Max tile:  min={df['max_tile'].min():>6}  mean={df['max_tile'].mean():>8.1f}  max={df['max_tile'].max():>6}")
    print(f"  Moves   :  min={df['move_count'].min():>6}  mean={df['move_count'].mean():>8.1f}  max={df['move_count'].max():>6}")
    win_rate = df["won"].mean() * 100
    print(f"  Win rate: {win_rate:.1f}%")
    print("=" * 50 + "\n")
