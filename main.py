"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--games N] [--algorithm random] [--output results/]

Examples
--------
    # Run 20 games with the random algorithm (default)
    python main.py --games 20

    # Show the browser window while playing
    python main.py --games 5 --show
"""

from __future__ import annotations

import argparse
import sys

from src.algorithms.random_algo import RandomAlgorithm
from src.runner import run_games
from src.visualize import plot_results

ALGORITHMS = {
    "random": RandomAlgorithm,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 2048 with algorithms via Playwright")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="random",
        help="Algorithm to use (default: random)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory for result charts (default: results/)",
    )
    parser.add_argument("--show", action="store_true", help="Show the browser window while playing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    algo_cls = ALGORITHMS[args.algorithm]
    algorithm = algo_cls()

    print(f"\nRunning {args.games} games with the '{algorithm.name}' algorithm…\n")
    df = run_games(algorithm, n_games=args.games, headless=not args.show)

    plot_results(df, output_dir=args.output)

    # Save raw data
    from pathlib import Path
    Path(args.output).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output) / f"results_{algorithm.name.lower()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved → {csv_path}")


if __name__ == "__main__":
    main()
