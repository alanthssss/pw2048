"""Run N games with a given algorithm and collect statistics."""

from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import List

import pandas as pd
from playwright.sync_api import sync_playwright

from src.game import Game2048
from src.algorithms.base import BaseAlgorithm

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclasses.dataclass
class GameResult:
    """Result of a single 2048 game."""

    run_id: str
    game_index: int
    algorithm: str
    score: int
    best_tile: int
    moves: int
    duration: float
    won: bool
    timestamp: str


def _run_batch(
    algo_cls: type,
    n_games: int,
    index_offset: int,
    headless: bool,
    move_delay_ms: int,
    run_id: str = "",
) -> list[dict]:
    """Run a batch of games inside a dedicated Playwright browser.

    This is a top-level function so it can be pickled by
    :class:`concurrent.futures.ProcessPoolExecutor`.  Each worker process
    creates its own Playwright context and browser instance.

    Parameters
    ----------
    algo_cls:
        Algorithm class to instantiate (must be importable from the worker).
    n_games:
        Number of games to play in this batch.
    index_offset:
        Starting game index (0-based); the first game gets ``index_offset + 1``.
    headless:
        Whether to launch Chromium in headless mode.
    move_delay_ms:
        Optional per-move delay in milliseconds.
    run_id:
        Identifier for this run (shared by all games in the batch).

    Returns
    -------
    list[dict]
        List of :class:`GameResult` dicts, one per game.
    """
    algorithm = algo_cls()
    results: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        game = Game2048.launch(page)

        for i in range(n_games):
            game.new_game()
            algorithm.on_game_start()
            t_start = time.perf_counter()

            while not game.is_game_over():
                board = game.get_board()
                direction = algorithm.choose_move(board)
                game.make_move(direction)
                if move_delay_ms:
                    page.wait_for_timeout(move_delay_ms)

            duration = time.perf_counter() - t_start
            ts = datetime.now(timezone.utc).strftime(_TS_FMT)
            result = dataclasses.asdict(
                GameResult(
                    run_id=run_id,
                    game_index=index_offset + i + 1,
                    algorithm=algorithm.name,
                    score=game.get_score(),
                    best_tile=game.get_max_tile(),
                    moves=game.get_move_count(),
                    duration=round(duration, 3),
                    won=game.is_won(),
                    timestamp=ts,
                )
            )
            results.append(result)
            print(
                f"  [worker offset={index_offset}] "
                f"Game {result['game_index']:>3}  "
                f"score={result['score']:>6}  "
                f"best_tile={result['best_tile']:>4}  "
                f"moves={result['moves']:>4}  "
                f"won={result['won']}"
            )

        browser.close()

    return results


def run_games(
    algorithm: BaseAlgorithm,
    n_games: int = 10,
    headless: bool = True,
    move_delay_ms: int = 0,
    n_workers: int = 1,
    run_id: str = "",
) -> pd.DataFrame:
    """
    Play *n_games* games using *algorithm* and return the results as a DataFrame.

    Parameters
    ----------
    algorithm : BaseAlgorithm
        The algorithm instance to use for all games.
    n_games : int
        Number of games to play.
    headless : bool
        Whether to run the browser in headless mode.
    move_delay_ms : int
        Optional delay in milliseconds between moves (useful for visual demos).
    n_workers : int
        Number of parallel browser workers.  When ``1`` (default) games run
        sequentially in the calling process.  When ``> 1`` games are split
        evenly across ``n_workers`` sub-processes, each with its own browser
        instance.  ``--show`` is silently forced to headless when
        ``n_workers > 1``.
    run_id : str
        Identifier for this run, stored in every result row.  Defaults to an
        empty string.

    Returns
    -------
    pd.DataFrame
        One row per game with columns: run_id, game_index, algorithm, score,
        best_tile, moves, duration, won, timestamp.
    """
    if n_workers <= 1:
        # ----------------------------------------------------------------
        # Sequential path (original behaviour, single browser)
        # ----------------------------------------------------------------
        results: List[GameResult] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            page = browser.new_page()
            game = Game2048.launch(page)

            for i in range(n_games):
                game.new_game()
                algorithm.on_game_start()
                t_start = time.perf_counter()

                while not game.is_game_over():
                    board = game.get_board()
                    direction = algorithm.choose_move(board)
                    game.make_move(direction)
                    if move_delay_ms:
                        page.wait_for_timeout(move_delay_ms)

                duration = time.perf_counter() - t_start
                ts = datetime.now(timezone.utc).strftime(_TS_FMT)
                results.append(
                    GameResult(
                        run_id=run_id,
                        game_index=i + 1,
                        algorithm=algorithm.name,
                        score=game.get_score(),
                        best_tile=game.get_max_tile(),
                        moves=game.get_move_count(),
                        duration=round(duration, 3),
                        won=game.is_won(),
                        timestamp=ts,
                    )
                )
                print(
                    f"  Game {i + 1:>3}/{n_games}  score={results[-1].score:>6}  "
                    f"best_tile={results[-1].best_tile:>4}  moves={results[-1].moves:>4}  "
                    f"won={results[-1].won}"
                )

            browser.close()

        return pd.DataFrame(dataclasses.asdict(r) for r in results)

    # --------------------------------------------------------------------
    # Parallel path – each worker gets its own browser
    # --------------------------------------------------------------------
    # Multi-window visual mode is impractical; silently force headless.
    effective_headless = True
    if not headless:
        print(
            "  [parallel] --show is not supported with multiple workers; "
            "running headless."
        )

    algo_cls = type(algorithm)

    # Distribute games as evenly as possible across workers.
    base = n_games // n_workers
    remainder = n_games % n_workers
    batches: list[tuple[int, int]] = []  # (count, offset)
    offset = 0
    for w in range(n_workers):
        count = base + (1 if w < remainder else 0)
        if count > 0:
            batches.append((count, offset))
            offset += count

    print(
        f"  [parallel] {n_games} games across {len(batches)} worker(s)…"
    )

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _run_batch, algo_cls, count, off, effective_headless, move_delay_ms, run_id
            )
            for count, off in batches
        ]
        for fut in futures:
            all_rows.extend(fut.result())

    all_rows.sort(key=lambda r: r["game_index"])
    return pd.DataFrame(all_rows)
