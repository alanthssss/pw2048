"""Greedy algorithm: pick the move that maximises immediate score gain."""

from __future__ import annotations

import random
from typing import List, Tuple

from .base import BaseAlgorithm
from src.game import DIRECTIONS


def _slide_row_left(row: List[int]) -> Tuple[List[int], int]:
    """Slide a single row left and return ``(new_row, score_gained)``."""
    tiles = [x for x in row if x != 0]
    score = 0
    merged: List[int] = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] * 2)
            score += tiles[i] * 2
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (4 - len(merged)))
    return merged, score


def simulate_move(
    board: List[List[int]], direction: str
) -> Tuple[List[List[int]], int]:
    """Simulate *direction* on *board* without modifying it.

    Returns the resulting board and the score gained.  If the move does not
    change the board the returned score will be 0.

    Parameters
    ----------
    board:
        4×4 matrix of tile values (0 = empty).
    direction:
        One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.
    """
    grid = [row[:] for row in board]

    if direction in ("left", "right"):
        total = 0
        for r in range(4):
            row = grid[r] if direction == "left" else grid[r][::-1]
            new_row, s = _slide_row_left(row)
            grid[r] = new_row if direction == "left" else new_row[::-1]
            total += s
        return grid, total

    # up / down — work column-wise via transposition
    transposed = [[board[r][c] for r in range(4)] for c in range(4)]
    total = 0
    for col in range(4):
        col_row = transposed[col] if direction == "up" else transposed[col][::-1]
        new_row, s = _slide_row_left(col_row)
        transposed[col] = new_row if direction == "up" else new_row[::-1]
        total += s
    new_grid = [[transposed[c][r] for c in range(4)] for r in range(4)]
    return new_grid, total


def _boards_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    return all(a[r][c] == b[r][c] for r in range(4) for c in range(4))


class GreedyAlgorithm(BaseAlgorithm):
    """Picks the move that yields the highest immediate score gain.

    For each candidate direction the move is simulated locally (no browser
    interaction).  The direction with the largest score gain is returned.
    Ties are broken by direction order (``"up"`` → ``"down"`` → ``"left"`` →
    ``"right"``).  When no move changes the board the algorithm falls back to
    a random direction.
    """

    name = "Greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the direction that maximises immediate score gain."""
        best_direction: str | None = None
        best_score = -1

        for direction in DIRECTIONS:
            new_board, score = simulate_move(board, direction)
            if not _boards_equal(board, new_board):
                if best_direction is None or score > best_score:
                    best_score = score
                    best_direction = direction

        return best_direction if best_direction is not None else self._rng.choice(DIRECTIONS)
