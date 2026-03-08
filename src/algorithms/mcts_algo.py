"""MCTS algorithm: Monte Carlo Tree Search for 2048."""

from __future__ import annotations

import math
import random
from typing import List

from .base import BaseAlgorithm
from .greedy_algo import simulate_move, _boards_equal
from src.game import DIRECTIONS

# Tile-spawn probabilities: 90 % chance of spawning a 2, 10 % chance of a 4.
_SPAWN_PROBS = ((2, 0.9), (4, 0.1))


def _get_empty_cells(board: List[List[int]]) -> list[tuple[int, int]]:
    """Return ``(row, col)`` pairs for every empty cell on the board."""
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def _spawn_tile(board: List[List[int]], rng: random.Random) -> List[List[int]]:
    """Return a copy of *board* with one new tile spawned at a random empty cell.

    Follows the 2048 spawn distribution: 90 % → 2, 10 % → 4.  If the board
    has no empty cells the board is returned unchanged.
    """
    empty = _get_empty_cells(board)
    if not empty:
        return [row[:] for row in board]
    new_board = [row[:] for row in board]
    r, c = rng.choice(empty)
    new_board[r][c] = 2 if rng.random() < 0.9 else 4
    return new_board


class _MCTSNode:
    """A node in the MCTS game tree.

    Each node represents the board state *after* a player move (and its
    subsequent tile spawn have been incorporated into the simulation).

    Attributes
    ----------
    board:
        Board state at this node.
    parent:
        Parent node, or ``None`` for the root.
    move:
        Direction chosen by the player to reach this node from the parent.
    children:
        Mapping from direction → child node for all expanded children.
    visits:
        Number of times this node has been visited.
    total_score:
        Cumulative rollout score accumulated across all visits.
    """

    __slots__ = ("board", "parent", "move", "children", "visits", "total_score", "_untried")

    def __init__(
        self,
        board: List[List[int]],
        parent: "_MCTSNode | None" = None,
        move: str | None = None,
    ) -> None:
        self.board = board
        self.parent = parent
        self.move = move
        self.children: dict[str, "_MCTSNode"] = {}
        self.visits: int = 0
        self.total_score: float = 0.0
        # Lazily computed list of unexpanded valid moves.
        self._untried: list[str] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_untried(self) -> list[str]:
        """Return valid directions not yet added as children."""
        return [
            d for d in DIRECTIONS
            if d not in self.children
            and not _boards_equal(self.board, simulate_move(self.board, d)[0])
        ]

    def untried_moves(self) -> list[str]:
        """Return (and cache) the list of unexpanded valid directions."""
        if self._untried is None:
            self._untried = self._compute_untried()
        return self._untried

    def is_fully_expanded(self) -> bool:
        """Return ``True`` when every valid direction has been expanded."""
        return len(self.untried_moves()) == 0

    def is_terminal(self) -> bool:
        """Return ``True`` when no valid move exists (game over)."""
        return all(
            _boards_equal(self.board, simulate_move(self.board, d)[0])
            for d in DIRECTIONS
        )

    def ucb1(self, exploration: float) -> float:
        """UCB1 score used during tree selection."""
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return (
            self.total_score / self.visits
            + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        )

    def best_child(self, exploration: float) -> "_MCTSNode":
        """Return the child with the highest UCB1 score."""
        return max(self.children.values(), key=lambda n: n.ucb1(exploration))

    def most_visited_child(self) -> "_MCTSNode":
        """Return the child with the highest visit count."""
        return max(self.children.values(), key=lambda n: n.visits)


class MCTSAlgorithm(BaseAlgorithm):
    """Monte Carlo Tree Search for 2048.

    Builds a search tree rooted at the current board state.  Each iteration
    of the algorithm performs four steps:

    1. **Selection** – traverse the tree from the root using the UCB1 formula
       until an unexpanded or terminal node is reached.
    2. **Expansion** – add one previously unseen move as a new child, spawning
       a random tile to produce the child's board state.
    3. **Simulation** – play random moves from the new child for up to
       *sim_depth* steps, accumulating the merge score.
    4. **Backpropagation** – propagate the simulation score back up to the
       root, updating visit counts and total scores along the way.

    After *n_iterations* iterations the move leading to the most-visited root
    child is returned.

    Parameters
    ----------
    n_iterations:
        Number of MCTS iterations (tree expansions) per move.  Higher values
        improve move quality but increase latency.
    sim_depth:
        Maximum number of random moves in each rollout simulation.
    exploration:
        UCB1 exploration constant *C*.  Higher values encourage broader
        exploration; lower values favour exploitation of known good moves.
    seed:
        Optional RNG seed for reproducibility.
    """

    name = "MCTS"
    version = "v2"

    def __init__(
        self,
        n_iterations: int = 400,
        sim_depth: int = 40,
        exploration: float = math.sqrt(2),
        seed: int | None = None,
    ) -> None:
        self._n_iterations = n_iterations
        self._sim_depth = sim_depth
        self._exploration = exploration
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_move(self, board: List[List[int]]) -> str:
        """Run MCTS and return the best direction found."""
        root = _MCTSNode(board)

        for _ in range(self._n_iterations):
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            score = self._simulate(node.board)
            self._backpropagate(node, score)

        if not root.children:
            # No valid move was ever expanded; fall back to random.
            return self._rng.choice(DIRECTIONS)

        return root.most_visited_child().move  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        """Traverse the tree to the most promising leaf using UCB1."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self._exploration)
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        """Add one untried move as a child node and return it."""
        untried = node.untried_moves()
        move = self._rng.choice(untried)
        untried.remove(move)

        new_board, _ = simulate_move(node.board, move)
        # Spawn a random tile to produce the child's board state.
        child_board = _spawn_tile(new_board, self._rng)
        child = _MCTSNode(child_board, parent=node, move=move)
        node.children[move] = child
        return child

    def _simulate(self, board: List[List[int]]) -> float:
        """Heuristic-guided rollout from *board* for up to *sim_depth* steps.

        At each step the algorithm evaluates all valid moves and picks the one
        with the highest immediate merge score (greedy exploitation).  Any valid
        move qualifies even if its merge score is zero (no merges), ensuring the
        rollout continues as long as moves exist.  Returns the total merge score
        accumulated during the rollout.
        """
        current = [row[:] for row in board]
        total_score = 0.0

        for _ in range(self._sim_depth):
            # Evaluate all valid moves; pick the one with the highest merge score.
            directions = list(DIRECTIONS)
            self._rng.shuffle(directions)

            best_board: list | None = None
            best_score: float = -1.0

            for d in directions:
                new_board, score = simulate_move(current, d)
                if not _boards_equal(current, new_board) and score > best_score:
                    best_score = score
                    best_board = new_board

            if best_board is None:
                break  # no valid move — game over

            current = _spawn_tile(best_board, self._rng)
            total_score += best_score

        return total_score

    def _backpropagate(self, node: _MCTSNode, score: float) -> None:
        """Propagate *score* from *node* up to the root."""
        current: _MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_score += score
            current = current.parent
