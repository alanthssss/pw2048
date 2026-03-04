"""Tests for the Random algorithm and game logic."""

from __future__ import annotations

import pathlib
import pytest
from playwright.sync_api import sync_playwright

from src.algorithms.random_algo import RandomAlgorithm
from src.game import Game2048, DIRECTIONS


GAME_URL = (pathlib.Path(__file__).parent.parent / "game.html").as_uri()


@pytest.fixture(scope="module")
def browser_page():
    """Provide a single browser page for all tests in this module."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


@pytest.fixture(scope="module")
def game(browser_page):
    """Launch the game and return a Game2048 instance."""
    return Game2048.launch(browser_page)


class TestRandomAlgorithm:
    def test_choose_move_returns_valid_direction(self):
        algo = RandomAlgorithm()
        board = [[0] * 4 for _ in range(4)]
        for _ in range(20):
            move = algo.choose_move(board)
            assert move in DIRECTIONS

    def test_seeded_rng_is_deterministic(self):
        algo1 = RandomAlgorithm(seed=42)
        algo2 = RandomAlgorithm(seed=42)
        board = [[0] * 4 for _ in range(4)]
        moves1 = [algo1.choose_move(board) for _ in range(10)]
        moves2 = [algo2.choose_move(board) for _ in range(10)]
        assert moves1 == moves2

    def test_algorithm_name(self):
        assert RandomAlgorithm.name == "Random"


class TestGame2048:
    def test_board_is_4x4(self, game):
        board = game.get_board()
        assert len(board) == 4
        assert all(len(row) == 4 for row in board)

    def test_initial_score_is_zero(self, game):
        game.new_game()
        assert game.get_score() == 0

    def test_initial_move_count_is_zero(self, game):
        game.new_game()
        assert game.get_move_count() == 0

    def test_game_is_not_over_at_start(self, game):
        game.new_game()
        assert not game.is_game_over()

    def test_max_tile_nonzero_at_start(self, game):
        game.new_game()
        assert game.get_max_tile() > 0

    def test_make_move_returns_bool(self, game):
        game.new_game()
        result = game.make_move("left")
        assert isinstance(result, bool)

    def test_invalid_direction_raises(self, game):
        with pytest.raises(ValueError):
            game.make_move("diagonal")

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the random algorithm and verify post-game state."""
        algo = RandomAlgorithm(seed=0)
        game.new_game()
        moves = 0
        while not game.is_game_over():
            board = game.get_board()
            direction = algo.choose_move(board)
            game.make_move(direction)
            moves += 1
            assert moves < 10_000, "Game did not end within 10 000 moves"

        assert game.get_score() >= 0
        assert game.get_max_tile() >= 2
        # move_count tracks effective (board-changing) moves; total attempts may be higher
        assert game.get_move_count() <= moves
