"""Abstract base class for 2048 algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseAlgorithm(ABC):
    """Abstract base for a 2048-playing algorithm.

    Subclasses implement :meth:`choose_move` which receives the current board
    state and returns a direction string.
    """

    #: Human-readable name shown in reports and charts.
    name: str = "Base"

    #: Version string included in per-run metadata.
    version: str = "v1"

    @abstractmethod
    def choose_move(self, board: List[List[int]]) -> str:
        """
        Choose the next move given the current board state.

        Parameters
        ----------
        board : list[list[int]]
            4×4 matrix of tile values (0 means empty).

        Returns
        -------
        str
            One of "up", "down", "left", "right".
        """
