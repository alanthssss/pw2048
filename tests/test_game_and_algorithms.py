"""Tests for the Random, Greedy, and Heuristic algorithms and game logic."""

from __future__ import annotations

import pathlib
import pytest
from playwright.sync_api import sync_playwright

from main import build_output_dir
from src.algorithms.greedy_algo import GreedyAlgorithm, simulate_move, _slide_row_left
from src.algorithms.heuristic_algo import (
    HeuristicAlgorithm,
    _corner_score,
    _empty_tiles_score,
    _merge_score,
    _monotonicity_score,
    _score_board,
)
from src.algorithms.random_algo import RandomAlgorithm
from src.game import Game2048, DIRECTIONS


GAME_URL = (pathlib.Path(__file__).parent.parent / "game.html").as_uri()



class TestBuildOutputDir:
    def test_returns_base_plus_algorithm_name(self):
        result = build_output_dir("results", "Random")
        assert result == pathlib.Path("results") / "Random"

    def test_algorithm_name_is_last_segment(self):
        result = build_output_dir("results", "MyAlgo")
        assert result.name == "MyAlgo"

    def test_custom_base_directory(self):
        result = build_output_dir("/tmp/custom", "Random")
        assert result == pathlib.Path("/tmp/custom") / "Random"

    def test_different_algorithm_names_produce_different_dirs(self):
        dir_a = build_output_dir("results", "AlgoA")
        dir_b = build_output_dir("results", "AlgoB")
        assert dir_a != dir_b


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


# ---------------------------------------------------------------------------
# _slide_row_left helper
# ---------------------------------------------------------------------------


class TestSlideRowLeft:
    def test_merges_equal_adjacent_tiles(self):
        row, score = _slide_row_left([2, 2, 0, 0])
        assert row == [4, 0, 0, 0]
        assert score == 4

    def test_slides_tiles_to_left(self):
        row, score = _slide_row_left([0, 0, 2, 4])
        assert row == [2, 4, 0, 0]
        assert score == 0

    def test_no_merge_different_tiles(self):
        row, score = _slide_row_left([2, 4, 2, 4])
        assert row == [2, 4, 2, 4]
        assert score == 0

    def test_double_merge(self):
        row, score = _slide_row_left([2, 2, 4, 4])
        assert row == [4, 8, 0, 0]
        assert score == 12

    def test_empty_row(self):
        row, score = _slide_row_left([0, 0, 0, 0])
        assert row == [0, 0, 0, 0]
        assert score == 0

    def test_single_merge_does_not_chain(self):
        # [2,2,2,0] → [4,2,0,0]: first pair merges, leftover 2 stays
        row, score = _slide_row_left([2, 2, 2, 0])
        assert row == [4, 2, 0, 0]
        assert score == 4


# ---------------------------------------------------------------------------
# simulate_move helper
# ---------------------------------------------------------------------------


class TestSimulateMove:
    def test_left_move(self):
        board = [
            [0, 2, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        new_board, score = simulate_move(board, "left")
        assert new_board[0] == [4, 0, 0, 0]
        assert score == 4

    def test_right_move(self):
        board = [
            [2, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        new_board, score = simulate_move(board, "right")
        assert new_board[0] == [0, 0, 0, 4]
        assert score == 4

    def test_up_move(self):
        board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
        ]
        new_board, score = simulate_move(board, "up")
        assert new_board[0][0] == 4
        assert score == 4

    def test_down_move(self):
        board = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        new_board, score = simulate_move(board, "down")
        assert new_board[3][0] == 4
        assert score == 4

    def test_does_not_mutate_original_board(self):
        board = [[2, 2, 0, 0]] + [[0] * 4 for _ in range(3)]
        original = [row[:] for row in board]
        simulate_move(board, "left")
        assert board == original


# ---------------------------------------------------------------------------
# GreedyAlgorithm
# ---------------------------------------------------------------------------


class TestGreedyAlgorithm:
    def test_choose_move_returns_valid_direction(self):
        algo = GreedyAlgorithm()
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_algorithm_name(self):
        assert GreedyAlgorithm.name == "Greedy"

    def test_prefers_merging_move(self):
        """Greedy should pick the direction that produces the highest merge score."""
        # Left produces a single merge (2+2=4, score 4); right produces nothing
        board = [
            [0, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        algo = GreedyAlgorithm()
        move = algo.choose_move(board)
        assert move == "left"

    def test_falls_back_when_no_valid_move_changes_board(self):
        """On a fully blocked board the fallback to a random direction works without error.

        The checkerboard layout below has no merges possible in any direction,
        so all simulated moves produce a board identical to the input.
        ``choose_move`` should detect that no direction changes the board and
        return a direction via its random fallback.
        """
        algo = GreedyAlgorithm(seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the greedy algorithm and verify post-game state."""
        algo = GreedyAlgorithm(seed=0)
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


# ---------------------------------------------------------------------------
# Heuristic helper functions
# ---------------------------------------------------------------------------


class TestEmptyTilesScore:
    def test_full_board_is_zero(self):
        board = [[2] * 4 for _ in range(4)]
        assert _empty_tiles_score(board) == 0

    def test_empty_board_is_sixteen(self):
        board = [[0] * 4 for _ in range(4)]
        assert _empty_tiles_score(board) == 16

    def test_partial_board(self):
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert _empty_tiles_score(board) == 15


class TestMonotonicityScore:
    def test_strictly_increasing_row_beats_random(self):
        ordered = [[1, 2, 4, 8], [0] * 4, [0] * 4, [0] * 4]
        shuffled = [[4, 1, 8, 2], [0] * 4, [0] * 4, [0] * 4]
        assert _monotonicity_score(ordered) >= _monotonicity_score(shuffled)

    def test_uniform_board_returns_zero(self):
        board = [[2] * 4 for _ in range(4)]
        assert _monotonicity_score(board) == 0


class TestCornerScore:
    def test_max_tile_in_corner_returns_max_tile(self):
        board = [[1024, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert _corner_score(board) == 1024

    def test_max_tile_not_in_corner_returns_zero(self):
        board = [[0, 1024, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert _corner_score(board) == 0

    def test_all_four_corners_recognized(self):
        for r, c in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            board = [[0] * 4 for _ in range(4)]
            board[r][c] = 512
            assert _corner_score(board) == 512


class TestMergeScore:
    def test_no_adjacent_equal_tiles(self):
        board = [[2, 4, 2, 4], [0] * 4, [0] * 4, [0] * 4]
        assert _merge_score(board) == 0

    def test_adjacent_equal_tiles_score(self):
        board = [[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]
        assert _merge_score(board) == 2

    def test_vertical_adjacent_equal_tiles(self):
        board = [[4, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert _merge_score(board) == 4


# ---------------------------------------------------------------------------
# HeuristicAlgorithm
# ---------------------------------------------------------------------------


class TestHeuristicAlgorithm:
    def test_algorithm_name(self):
        assert HeuristicAlgorithm.name == "Heuristic"

    def test_choose_move_returns_valid_direction(self):
        algo = HeuristicAlgorithm()
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_prefers_move_that_puts_max_tile_in_corner(self):
        """Moving left is the only move that places the max tile in a corner."""
        # 1024 is at [0][2]; moving left → [0][0] (corner, score boost).
        # Moving right keeps 1024 at [0][2] (no change, invalid move).
        # Moving down → 1024 at [3][2] (not a corner).
        # Moving up → 1024 stays at [0][2] (no change, invalid move).
        board = [
            [0, 0, 1024, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        algo = HeuristicAlgorithm()
        move = algo.choose_move(board)
        assert move == "left"

    def test_falls_back_on_fully_blocked_board(self):
        """On a checkerboard with no valid moves the random fallback is used."""
        algo = HeuristicAlgorithm(seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_seeded_rng_fallback_is_deterministic(self):
        """Two instances with the same seed produce the same fallback direction."""
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move1 = HeuristicAlgorithm(seed=7).choose_move(board)
        move2 = HeuristicAlgorithm(seed=7).choose_move(board)
        assert move1 == move2

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the heuristic algorithm and verify post-game state."""
        algo = HeuristicAlgorithm(seed=0)
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
