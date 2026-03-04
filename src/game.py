"""Playwright wrapper for the local 2048 game."""

from __future__ import annotations

import pathlib
from typing import List

from playwright.sync_api import Page


GAME_URL = pathlib.Path(__file__).parent.parent / "game.html"
DIRECTIONS = ["up", "down", "left", "right"]


class Game2048:
    """Controls a 2048 game running in a Playwright page."""

    def __init__(self, page: Page) -> None:
        self.page = page

    @classmethod
    def launch(cls, page: Page) -> "Game2048":
        """Navigate to the local game HTML and return a ready Game2048 instance."""
        page.goto(GAME_URL.as_uri())
        page.wait_for_function("typeof window.isGameOver === 'function'")
        return cls(page)

    def new_game(self) -> None:
        """Start a new game."""
        self.page.evaluate("window.newGame()")

    def get_board(self) -> List[List[int]]:
        """Return the current 4×4 board as a list of lists."""
        return self.page.evaluate("window.getBoard()")

    def get_score(self) -> int:
        """Return the current score."""
        return self.page.evaluate("window.getScore()")

    def get_move_count(self) -> int:
        """Return the number of moves made so far."""
        return self.page.evaluate("window.getMoveCount()")

    def get_max_tile(self) -> int:
        """Return the highest tile value currently on the board."""
        return self.page.evaluate("window.getMaxTile()")

    def is_game_over(self) -> bool:
        """Return True when no moves remain."""
        return self.page.evaluate("window.isGameOver()")

    def is_won(self) -> bool:
        """Return True when the 2048 tile has been reached."""
        return self.page.evaluate("window.isWon()")

    def make_move(self, direction: str) -> bool:
        """
        Execute a move in the given direction.

        Parameters
        ----------
        direction : str
            One of "up", "down", "left", "right".

        Returns
        -------
        bool
            True if the move changed the board, False otherwise.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        return self.page.evaluate(f"window.makeMove('{direction}')")
