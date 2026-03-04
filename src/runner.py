"""Run N games with a given algorithm and collect statistics."""

from __future__ import annotations

import dataclasses
import time
from typing import List

import pandas as pd
from playwright.sync_api import sync_playwright

from src.game import Game2048
from src.algorithms.base import BaseAlgorithm


@dataclasses.dataclass
class GameResult:
    """Result of a single 2048 game."""

    game_index: int
    score: int
    max_tile: int
    move_count: int
    won: bool
    duration_s: float
    algorithm: str


def run_games(
    algorithm: BaseAlgorithm,
    n_games: int = 10,
    headless: bool = True,
    move_delay_ms: int = 0,
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

    Returns
    -------
    pd.DataFrame
        One row per game with columns: game_index, score, max_tile,
        move_count, won, duration_s, algorithm.
    """
    results: List[GameResult] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        game = Game2048.launch(page)

        for i in range(n_games):
            game.new_game()
            t_start = time.perf_counter()

            while not game.is_game_over():
                board = game.get_board()
                direction = algorithm.choose_move(board)
                game.make_move(direction)
                if move_delay_ms:
                    page.wait_for_timeout(move_delay_ms)

            duration = time.perf_counter() - t_start
            results.append(
                GameResult(
                    game_index=i + 1,
                    score=game.get_score(),
                    max_tile=game.get_max_tile(),
                    move_count=game.get_move_count(),
                    won=game.is_won(),
                    duration_s=round(duration, 3),
                    algorithm=algorithm.name,
                )
            )
            print(
                f"  Game {i + 1:>3}/{n_games}  score={results[-1].score:>6}  "
                f"max_tile={results[-1].max_tile:>4}  moves={results[-1].move_count:>4}  "
                f"won={results[-1].won}"
            )

        browser.close()

    return pd.DataFrame(dataclasses.asdict(r) for r in results)
