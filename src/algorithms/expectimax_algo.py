"""Expectimax algorithm: game-tree search with chance nodes for tile spawns."""

from __future__ import annotations

import random
from typing import List

from .base import BaseAlgorithm
from .greedy_algo import simulate_move, _boards_equal
from .heuristic_algo import _score_board
from src.game import DIRECTIONS

# Tile-spawn probabilities used by the 2048 game: 90 % → 2, 10 % → 4.
_SPAWN_PROBS = ((2, 0.9), (4, 0.1))

# Maximum number of empty cells to sample at each chance node.  Capping this
# value keeps the per-move search time reasonable when many cells are free
# (early game) without noticeably weakening play.
_MAX_CHANCE_CELLS = 8


def _get_empty_cells(board: List[List[int]]) -> list[tuple[int, int]]:
    """Return a list of ``(row, col)`` positions for every empty cell."""
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def _expectimax(board: List[List[int]], depth: int, is_player: bool) -> float:
    """Recursive expectimax search.

    Alternates between *player* (max) nodes and *chance* nodes.  The depth
    counter decreases by one at each node; when it reaches zero the board is
    evaluated with the heuristic score function.

    Parameters
    ----------
    board:
        Current 4×4 board state.
    depth:
        Remaining half-steps.  Depth 0 triggers leaf evaluation.
    is_player:
        ``True`` for a max (player) node; ``False`` for a chance (tile-spawn)
        node.
    """
    if depth == 0:
        return _score_board(board)

    if is_player:
        best = float("-inf")
        any_valid = False
        for direction in DIRECTIONS:
            new_board, _ = simulate_move(board, direction)
            if not _boards_equal(board, new_board):
                any_valid = True
                val = _expectimax(new_board, depth - 1, False)
                if val > best:
                    best = val
        # No valid moves → game over; evaluate current state.
        return best if any_valid else _score_board(board)

    # Chance node: average over all possible tile placements.
    empty = _get_empty_cells(board)
    if not empty:
        return _score_board(board)

    # Sample at most _MAX_CHANCE_CELLS to keep the branching factor bounded.
    if len(empty) > _MAX_CHANCE_CELLS:
        # Use the first _MAX_CHANCE_CELLS cells (deterministic, no RNG needed here).
        empty = empty[:_MAX_CHANCE_CELLS]

    total = 0.0
    for r, c in empty:
        for value, prob in _SPAWN_PROBS:
            new_board = [row[:] for row in board]
            new_board[r][c] = value
            total += prob * _expectimax(new_board, depth - 1, True)
    return total / len(empty)


class ExpectimaxAlgorithm(BaseAlgorithm):
    """Game-tree search with chance nodes for tile spawns.

    The search tree alternates between *player* (max) nodes, where the best
    direction is chosen, and *chance* nodes, where the expected value across
    all possible tile spawns is computed.  Leaf nodes are evaluated with the
    same heuristic used by :class:`~src.algorithms.heuristic_algo.HeuristicAlgorithm`.

    Parameters
    ----------
    depth:
        Search depth measured in *half-steps* (a player turn plus its
        following chance node counts as two half-steps).  The default of 4
        gives two full player-turn look-ahead cycles.  Increasing this value
        improves play strength at the cost of higher per-move latency.
    seed:
        Optional RNG seed for the random fallback used when no valid move
        exists (i.e. the game is already over).
    """

    name = "Expectimax"

    def __init__(self, depth: int = 4, seed: int | None = None) -> None:
        self._depth = depth
        self._rng = random.Random(seed)

    def choose_move(self, board: List[List[int]]) -> str:
        """Return the direction with the highest expectimax value."""
        best_direction: str | None = None
        best_score = float("-inf")

        for direction in DIRECTIONS:
            new_board, _ = simulate_move(board, direction)
            if not _boards_equal(board, new_board):
                # After the player's first move the next node is a chance node.
                val = _expectimax(new_board, self._depth - 1, False)
                if val > best_score:
                    best_score = val
                    best_direction = direction

        return best_direction if best_direction is not None else self._rng.choice(DIRECTIONS)
