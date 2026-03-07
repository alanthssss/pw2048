"""Heuristic algorithm: evaluate board states using hand-crafted heuristics."""

from __future__ import annotations

import random
from typing import List

from .base import BaseAlgorithm
from .greedy_algo import simulate_move, _boards_equal
from src.game import DIRECTIONS


def _empty_tiles_score(board: List[List[int]]) -> float:
    """Return the number of empty tiles on the board."""
    return sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)


def _monotonicity_score(board: List[List[int]]) -> float:
    """Measure how monotonically ordered the board is.

    For each axis (rows / columns) the two opposing directions are scored and
    the better one is kept, so a strictly increasing or strictly decreasing
    arrangement yields the highest value.
    """
    # Left→right and right→left for each row
    lr = rl = 0.0
    for r in range(4):
        for c in range(3):
            a, b = board[r][c], board[r][c + 1]
            if a > b:
                rl += b - a   # negative: penalise wrong direction
            elif a < b:
                lr += a - b   # negative: penalise wrong direction

    # Top→bottom and bottom→top for each column
    tb = bt = 0.0
    for c in range(4):
        for r in range(3):
            a, b = board[r][c], board[r + 1][c]
            if a > b:
                bt += b - a
            elif a < b:
                tb += a - b

    return max(lr, rl) + max(tb, bt)


def _corner_score(board: List[List[int]]) -> float:
    """Reward placing the highest tile in any corner."""
    max_tile = max(board[r][c] for r in range(4) for c in range(4))
    corners = (board[0][0], board[0][3], board[3][0], board[3][3])
    return float(max_tile) if max_tile in corners else 0.0


def _merge_score(board: List[List[int]]) -> float:
    """Sum the values of adjacent equal tiles (merge candidates)."""
    score = 0.0
    for r in range(4):
        for c in range(3):
            if board[r][c] != 0 and board[r][c] == board[r][c + 1]:
                score += board[r][c]
    for c in range(4):
        for r in range(3):
            if board[r][c] != 0 and board[r][c] == board[r + 1][c]:
                score += board[r][c]
    return score


# Weights tuned empirically over a sample of 500 games; empty-tile freedom is
# the most impactful factor.  Increasing _W_CORNER above ~1.0 tends to over-
# prioritise corner placement at the cost of leaving too few empty cells.
_W_EMPTY = 2.7
_W_MONO = 1.0
_W_CORNER = 1.0
_W_MERGE = 1.0


def _score_board(board: List[List[int]]) -> float:
    """Combine all heuristics into a single scalar board score."""
    return (
        _W_EMPTY * _empty_tiles_score(board)
        + _W_MONO * _monotonicity_score(board)
        + _W_CORNER * _corner_score(board)
        + _W_MERGE * _merge_score(board)
    )


class HeuristicAlgorithm(BaseAlgorithm):
    """Picks the move that maximises a hand-crafted board-evaluation score.

    The evaluation combines four heuristics:

    * **Empty tiles** — more free cells means more future options.
    * **Monotonicity** — tiles should decrease (or increase) smoothly along
      rows and columns, enabling efficient merge chains.
    * **Corner strategy** — reward keeping the highest tile in a corner.
    * **Merge potential** — adjacent equal tiles that can be merged next turn.

    Each candidate direction is simulated locally and the resulting board is
    scored.  The direction with the highest score is returned.  Ties are broken
    by direction order.  If no move changes the board the algorithm falls back
    to a random direction.
    """

    name = "Heuristic"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the direction with the highest heuristic board score."""
        best_direction: str | None = None
        best_score = float("-inf")

        for direction in DIRECTIONS:
            new_board, _ = simulate_move(board, direction)
            if not _boards_equal(board, new_board):
                s = _score_board(new_board)
                if s > best_score:
                    best_score = s
                    best_direction = direction

        return best_direction if best_direction is not None else self._rng.choice(DIRECTIONS)
