"""Simple random algorithm: pick a random direction each turn."""

from __future__ import annotations

import random
from typing import List

from .base import BaseAlgorithm
from src.game import DIRECTIONS


class RandomAlgorithm(BaseAlgorithm):
    """Selects a uniformly random direction on every move."""

    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: List[List[int]]) -> str:
        """Return a random direction regardless of board state."""
        return self._rng.choice(DIRECTIONS)
