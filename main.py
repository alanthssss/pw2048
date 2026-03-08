# PYTHON_ARGCOMPLETE_OK
"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--tui | --gui | --web]
                   [--mode dev|release|benchmark]
                   [--games N] [--runs N] [--algorithm random] [--output results/]
                   [--keep N] [--report] [--parallel N]
                   [--s3-bucket BUCKET] [--s3-prefix PREFIX] [--s3-public]

    Available algorithms: random, greedy, heuristic, expectimax,
                          mcts-v1, mcts-v2, mcts,
                          dqn-v1, dqn-v2, dqn,
                          ppo-v1, ppo-v2, ppo

    Results are saved under a per-run subdirectory, e.g.::

        results/Random/run_20260307_120000/results.csv
        results/Random/run_20260307_120000/chart.png
        results/Random/run_20260307_120000/metrics.json

    ``--mode``       sets games/runs/parallel automatically:
                     dev: 100 games, 1 run, parallel=auto
                     release: 1000 games, 1 run, parallel=auto
                     benchmark: 500 games, 5 runs, parallel=auto

    ``--keep N``     keeps only the *N* most-recent runs per algorithm (locally
    and, when ``--s3-bucket`` is given, on S3).  Defaults to 10.

    ``--report``     generates a self-contained HTML dashboard (``index.html``)
    in the output directory.  When ``--s3-bucket`` is given the report is also
    uploaded to ``<s3-prefix>/index.html``.

    ``--parallel N`` launches *N* browser workers simultaneously, each playing
    its share of the games.  Defaults to 1 (sequential).

Examples
--------
    # Run 20 games with the random algorithm (default)
    python main.py --games 20

    # Quick dev scratch run (100 games, auto-parallel)
    python main.py --mode dev

    # Benchmark run (500 games × 5 runs, auto-parallel)
    python main.py --mode benchmark --report

    # Show the browser window while playing
    python main.py --games 5 --show

    # Keep the 5 most-recent runs and generate an HTML report
    python main.py --games 10 --keep 5 --report

    # Run 40 games using 4 parallel browser workers
    python main.py --games 40 --parallel 4 --report

    # Upload results to S3
    python main.py --games 10 --s3-bucket my-bucket --report

    # Launch the interactive TUI wizard
    python main.py --tui

    # Launch the desktop GUI wizard
    python main.py --gui

    # Launch the web UI wizard in the browser
    python main.py --web
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import argcomplete

from src.algorithms.dqn_algo import DQNAlgorithmV1, DQNAlgorithmV2
from src.algorithms.expectimax_algo import ExpectimaxAlgorithm
from src.algorithms.greedy_algo import GreedyAlgorithm
from src.algorithms.heuristic_algo import HeuristicAlgorithm
from src.algorithms.mcts_algo import MCTSAlgorithmV1, MCTSAlgorithmV2
from src.algorithms.ppo_algo import PPOAlgorithmV1, PPOAlgorithmV2
from src.algorithms.random_algo import RandomAlgorithm
from src.runner import run_games
from src.visualize import plot_results

ALGORITHMS = {
    "random":     RandomAlgorithm,
    "greedy":     GreedyAlgorithm,
    "heuristic":  HeuristicAlgorithm,
    "expectimax": ExpectimaxAlgorithm,
    # MCTS — both versions registered; "mcts" is an alias for the latest.
    "mcts-v1":    MCTSAlgorithmV1,
    "mcts-v2":    MCTSAlgorithmV2,
    "mcts":       MCTSAlgorithmV2,
    # DQN — both versions registered; "dqn" is an alias for the latest.
    "dqn-v1":     DQNAlgorithmV1,
    "dqn-v2":     DQNAlgorithmV2,
    "dqn":        DQNAlgorithmV2,
    # PPO — both versions registered; "ppo" is an alias for the latest.
    "ppo-v1":     PPOAlgorithmV1,
    "ppo-v2":     PPOAlgorithmV2,
    "ppo":        PPOAlgorithmV2,
}

_DEFAULT_KEEP = 10
_DEFAULT_S3_PREFIX = "results"

# Mode presets: games, runs, parallel ("auto" → os.cpu_count())
_MODE_PRESETS: dict[str, dict] = {
    "dev":       {"games": 100,  "runs": 1, "parallel": os.cpu_count() or 1},
    "release":   {"games": 1000, "runs": 1, "parallel": os.cpu_count() or 1},
    "benchmark": {"games": 500,  "runs": 5, "parallel": os.cpu_count() or 1},
}
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1


def build_output_dir(base: str | Path, algorithm_name: str) -> Path:
    """Return the algorithm-scoped output directory.

    The resulting path has the form::

        <base>/<AlgorithmName>/

    Individual runs are stored in ``run_<timestamp>/`` subdirectories under
    this directory (see :func:`build_run_dir`).

    Parameters
    ----------
    base:
        Root directory for all results (e.g. ``"results"``).
    algorithm_name:
        Human-readable algorithm name (used verbatim as a subdirectory).
    """
    return Path(base) / algorithm_name


def build_run_dir(algo_dir: Path, run_id: str) -> Path:
    """Return the per-run subdirectory path: ``<algo_dir>/run_<run_id>/``.

    Parameters
    ----------
    algo_dir:
        Algorithm-scoped output directory returned by :func:`build_output_dir`.
    run_id:
        Timestamp string identifying this run (e.g. ``"20260307_120000"``).
    """
    return algo_dir / f"run_{run_id}"


def _get_git_commit() -> str:
    """Return the short HEAD commit hash, or ``"unknown"`` when not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_run_metadata(
    run_dir: Path,
    algorithm_name: str,
    algorithm_version: str,
    n_games: int,
    n_workers: int,
    timestamp: str,
    mode: str | None,
) -> Path:
    """Write ``metrics.json`` to *run_dir* and return its path.

    Parameters
    ----------
    run_dir:
        Directory for this run (already created).
    algorithm_name:
        Human-readable algorithm name.
    algorithm_version:
        Version string from the algorithm class.
    n_games:
        Number of games played in this run.
    n_workers:
        Number of parallel browser workers used.
    timestamp:
        ISO 8601 UTC timestamp string for this run.
    mode:
        The ``--mode`` value (``"dev"``, ``"release"``, ``"benchmark"``) or
        ``"custom"`` when no mode flag was given.

    Returns
    -------
    Path
        Path to the written ``metrics.json`` file.
    """
    meta = {
        "algorithm":         algorithm_name,
        "algorithm_version": algorithm_version,
        "games":             n_games,
        "parallel_workers":  n_workers,
        "timestamp":         timestamp,
        "git_commit":        _get_git_commit(),
        "mode":              mode or "custom",
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def prune_local_results(output_dir: Path, keep_n: int) -> list[Path]:
    """Delete old run directories, keeping only the latest *keep_n* runs.

    Each "run" is a subdirectory named ``run_<timestamp>`` under *output_dir*.
    Directories are sorted lexicographically (chronological order because the
    timestamp prefix is ``YYYYMMDD_HHMMSS``).

    Parameters
    ----------
    output_dir:
        Algorithm-scoped results directory (e.g. ``results/Random/``).
    keep_n:
        Number of most-recent run directories to retain.  Pass ``0`` to keep all.

    Returns
    -------
    list[Path]
        Paths of the deleted files.
    """
    if keep_n <= 0:
        return []
    run_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ) if output_dir.exists() else []
    old_dirs = run_dirs[:-keep_n] if len(run_dirs) > keep_n else []

    deleted: list[Path] = []
    for d in old_dirs:
        for f in sorted(d.iterdir()):
            f.unlink()
            deleted.append(f)
        d.rmdir()
        deleted.append(d)
    if deleted:
        print(f"  Pruned {len(old_dirs)} old run dir(s) in '{output_dir}'")
    return deleted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 2048 with algorithms via Playwright")
    parser.add_argument(
        "--mode",
        choices=["dev", "release", "benchmark"],
        default=None,
        help=(
            "Run mode preset — sets games/runs/parallel automatically. "
            "dev: 100 games, 1 run; release: 1000 games, 1 run; "
            "benchmark: 500 games, 5 runs. "
            "Explicit --games/--runs/--parallel flags override the preset."
        ),
    )
    parser.add_argument("--games", type=int, default=None, help="Number of games per run (default: 20)")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        metavar="N",
        help="Number of times to repeat the full set of games (default: 1). "
             "Each run produces its own run_<timestamp>/ directory.",
    )
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="random",
        help="Algorithm to use (default: random)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Base directory for result charts (default: results/). "
             "Results are saved under <output>/<algorithm>/run_<timestamp>/",
    )
    parser.add_argument("--show", action="store_true", help="Show the browser window while playing")
    parser.add_argument(
        "--keep",
        type=int,
        default=_DEFAULT_KEEP,
        metavar="N",
        help=f"Keep only the N most-recent runs per algorithm (default: {_DEFAULT_KEEP}). "
             "Pass 0 to disable pruning.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel browser workers (default: 1 = sequential). "
             "Each worker runs its share of games in a separate browser instance.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a self-contained HTML results dashboard (index.html) in the output directory.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        metavar="BUCKET",
        help="S3 bucket to upload results and the HTML report to.  "
             "Requires boto3 and valid AWS credentials.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=_DEFAULT_S3_PREFIX,
        metavar="PREFIX",
        help=f"Key prefix inside the S3 bucket (default: '{_DEFAULT_S3_PREFIX}').",
    )
    parser.add_argument(
        "--s3-public",
        action="store_true",
        help="Apply public-read ACL to uploaded S3 objects.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive TUI wizard to configure and start a run.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI wizard (tkinter) to configure and start a run.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open the web UI launcher in the system browser to configure and start a run.",
    )
    parser.add_argument(
        "--algo-version",
        default=None,
        metavar="VERSION",
        help="Override the algorithm version tag written to metrics.json "
             "(e.g. 'v2-experiment').  Defaults to the algorithm class's own "
             "'version' attribute.",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    # Apply mode presets for any value not explicitly provided on the CLI.
    if args.mode is not None:
        preset = _MODE_PRESETS[args.mode]
        if args.games is None:
            args.games = preset["games"]
        if args.runs is None:
            args.runs = preset["runs"]
        if args.parallel is None:
            args.parallel = preset["parallel"]

    # Fall back to plain defaults when no mode was given.
    if args.games is None:
        args.games = _DEFAULT_GAMES
    if args.runs is None:
        args.runs = _DEFAULT_RUNS
    if args.parallel is None:
        args.parallel = _DEFAULT_PARALLEL

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Launch the interactive TUI wizard when --tui is requested, then
    # re-parse the argv it returns.
    if args.tui:
        from src.tui import run_tui

        tui_argv = run_tui()
        args = parse_args(tui_argv)

    # Launch the desktop GUI wizard when --gui is requested.
    if args.gui:
        from src.gui import run_gui

        gui_argv = run_gui()
        args = parse_args(gui_argv)

    # Launch the web UI when --web is requested.
    if args.web:
        from src.webui import run_webui

        web_argv = run_webui()
        args = parse_args(web_argv)

    algo_cls = ALGORITHMS[args.algorithm]
    algorithm = algo_cls()

    algo_dir = build_output_dir(args.output, algorithm.name)
    algo_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, args.parallel)
    n_runs = max(1, args.runs)
    parallel_note = f" ({n_workers} parallel workers)" if n_workers > 1 else ""
    runs_note = f" × {n_runs} run(s)" if n_runs > 1 else ""
    print(
        f"\nRunning {args.games} games{runs_note} with the '{algorithm.name}' algorithm{parallel_note}…\n"
    )

    last_run_id = None
    for run_num in range(1, n_runs + 1):
        if n_runs > 1:
            print(f"\n─── Run {run_num}/{n_runs} ───")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_run_id = run_id
        run_dir = build_run_dir(algo_dir, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        df = run_games(
            algorithm,
            n_games=args.games,
            headless=not args.show,
            n_workers=n_workers,
            run_id=run_id,
        )

        # Save chart (always named chart.png inside the run dir)
        plot_results(df, output_dir=run_dir, output_stem="chart")

        # Save raw data
        csv_path = run_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Raw data saved → {csv_path}")

        # Save run metadata
        ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # --algo-version overrides the class-level version; fall back to the
        # algorithm's own 'version' attribute (or "v1" for legacy classes).
        effective_version = (
            args.algo_version.strip()
            if args.algo_version and args.algo_version.strip()
            else getattr(algorithm, "version", "v1")
        )
        meta_path = write_run_metadata(
            run_dir=run_dir,
            algorithm_name=algorithm.name,
            algorithm_version=effective_version,
            n_games=args.games,
            n_workers=n_workers,
            timestamp=ts_iso,
            mode=args.mode,
        )
        print(f"Metadata saved → {meta_path}")

        # ------------------------------------------------------------------
        # Prune old local results after each run to keep disk usage bounded
        # ------------------------------------------------------------------
        prune_local_results(algo_dir, keep_n=args.keep)

    # ------------------------------------------------------------------
    # Generate HTML report
    # ------------------------------------------------------------------
    if args.report or args.s3_bucket:
        from src.report import generate_html_report

        report_path = Path(args.output) / "index.html"
        generate_html_report(results_dir=args.output, output_path=report_path)
        print(f"Report saved  → {report_path}")

    # ------------------------------------------------------------------
    # Upload to S3
    # ------------------------------------------------------------------
    if args.s3_bucket:
        from src.storage import sync_run_to_s3, upload_report

        print(f"\nUploading results to s3://{args.s3_bucket}/{args.s3_prefix}/…")
        sync_run_to_s3(
            algo_dir=algo_dir,
            run_id=last_run_id,
            bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            algorithm_name=algorithm.name,
            keep_n=args.keep,
            public_read=args.s3_public,
        )

        report_path = Path(args.output) / "index.html"
        if report_path.exists():
            upload_report(
                report_path=report_path,
                bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                public_read=args.s3_public,
            )


if __name__ == "__main__":
    main()
