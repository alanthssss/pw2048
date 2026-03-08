"""Tests for the Random, Greedy, Heuristic, Expectimax, MCTS, DQN, and PPO algorithms and game logic."""

from __future__ import annotations

import pathlib
import random
import pytest
from playwright.sync_api import sync_playwright

from main import build_output_dir
from src.algorithms.dqn_algo import DQNAlgorithm, _encode_board, _board_heuristic, _QNetwork
from src.algorithms.expectimax_algo import ExpectimaxAlgorithm, _expectimax, _get_empty_cells
from src.algorithms.greedy_algo import GreedyAlgorithm, simulate_move, _slide_row_left, _boards_equal
from src.algorithms.heuristic_algo import (
    HeuristicAlgorithm,
    _corner_score,
    _empty_tiles_score,
    _merge_score,
    _monotonicity_score,
    _score_board,
)
from src.algorithms.mcts_algo import MCTSAlgorithm, _MCTSNode, _spawn_tile
from src.algorithms.ppo_algo import PPOAlgorithm
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


# ---------------------------------------------------------------------------
# Expectimax helper functions
# ---------------------------------------------------------------------------


class TestGetEmptyCells:
    def test_full_board_returns_empty_list(self):
        board = [[2] * 4 for _ in range(4)]
        assert _get_empty_cells(board) == []

    def test_empty_board_returns_all_cells(self):
        board = [[0] * 4 for _ in range(4)]
        assert len(_get_empty_cells(board)) == 16

    def test_partial_board(self):
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert len(_get_empty_cells(board)) == 15
        assert (0, 0) not in _get_empty_cells(board)


class TestExpectimaxSearch:
    def test_depth_zero_returns_heuristic_score(self):
        board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        val = _expectimax(board, 0, True)
        assert val == _score_board(board)

    def test_player_node_returns_finite_value(self):
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        val = _expectimax(board, 2, True)
        assert isinstance(val, float)

    def test_chance_node_returns_finite_value(self):
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        val = _expectimax(board, 2, False)
        assert isinstance(val, float)

    def test_terminal_board_returns_heuristic_score(self):
        # Checkerboard: no valid moves → should return the heuristic score.
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        val = _expectimax(board, 4, True)
        assert val == _score_board(board)


# ---------------------------------------------------------------------------
# ExpectimaxAlgorithm
# ---------------------------------------------------------------------------


class TestExpectimaxAlgorithm:
    def test_algorithm_name(self):
        assert ExpectimaxAlgorithm.name == "Expectimax"

    def test_choose_move_returns_valid_direction(self):
        algo = ExpectimaxAlgorithm()
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_falls_back_on_fully_blocked_board(self):
        algo = ExpectimaxAlgorithm(seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_seeded_rng_fallback_is_deterministic(self):
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move1 = ExpectimaxAlgorithm(seed=7).choose_move(board)
        move2 = ExpectimaxAlgorithm(seed=7).choose_move(board)
        assert move1 == move2

    def test_prefers_corner_move(self):
        """Moving left should place the max tile in the top-left corner."""
        board = [
            [0, 0, 1024, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        algo = ExpectimaxAlgorithm(depth=2)
        move = algo.choose_move(board)
        assert move == "left"

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the expectimax algorithm."""
        algo = ExpectimaxAlgorithm(depth=2, seed=0)
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
# MCTS helpers
# ---------------------------------------------------------------------------


class TestSpawnTile:
    def test_adds_a_tile_to_empty_cell(self):
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        rng = random.Random(0)
        new_board = _spawn_tile(board, rng)
        # Original board is unchanged.
        assert board[0] == [2, 0, 0, 0]
        # Exactly one more non-zero tile.
        orig_nonzero = sum(board[r][c] != 0 for r in range(4) for c in range(4))
        new_nonzero = sum(new_board[r][c] != 0 for r in range(4) for c in range(4))
        assert new_nonzero == orig_nonzero + 1

    def test_spawned_tile_is_2_or_4(self):
        board = [[0] * 4 for _ in range(4)]
        rng = random.Random(42)
        for _ in range(20):
            nb = _spawn_tile(board, rng)
            spawned = {nb[r][c] for r in range(4) for c in range(4)} - {0}
            assert spawned <= {2, 4}

    def test_full_board_unchanged(self):
        board = [[2] * 4 for _ in range(4)]
        rng = random.Random(0)
        new_board = _spawn_tile(board, rng)
        assert new_board == board


class TestMCTSNode:
    def _simple_board(self):
        return [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def test_untried_moves_excludes_invalid_directions(self):
        node = _MCTSNode(self._simple_board())
        untried = node.untried_moves()
        # All returned moves must be valid (change the board).
        for d in untried:
            new_b, _ = simulate_move(node.board, d)
            assert not _boards_equal(node.board, new_b)

    def test_is_terminal_false_for_open_board(self):
        node = _MCTSNode(self._simple_board())
        assert not node.is_terminal()

    def test_is_terminal_true_for_blocked_board(self):
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        node = _MCTSNode(board)
        assert node.is_terminal()

    def test_ucb1_infinite_for_unvisited(self):
        parent = _MCTSNode([[0] * 4 for _ in range(4)])
        parent.visits = 1
        child = _MCTSNode([[0] * 4 for _ in range(4)], parent=parent, move="left")
        assert child.ucb1(1.0) == float("inf")

    def test_most_visited_child_returns_highest_visits(self):
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        root = _MCTSNode(board)
        for move, visits in [("left", 5), ("right", 10), ("up", 1)]:
            child = _MCTSNode(board, parent=root, move=move)
            child.visits = visits
            root.children[move] = child
        assert root.most_visited_child().move == "right"


# ---------------------------------------------------------------------------
# MCTSAlgorithm
# ---------------------------------------------------------------------------


class TestMCTSAlgorithm:
    def test_algorithm_name(self):
        assert MCTSAlgorithm.name == "MCTS"

    def test_choose_move_returns_valid_direction(self):
        algo = MCTSAlgorithm(n_iterations=20, seed=0)
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_falls_back_on_fully_blocked_board(self):
        algo = MCTSAlgorithm(n_iterations=20, seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_seeded_instance_is_deterministic(self):
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move1 = MCTSAlgorithm(n_iterations=30, seed=42).choose_move(board)
        move2 = MCTSAlgorithm(n_iterations=30, seed=42).choose_move(board)
        assert move1 == move2

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the MCTS algorithm."""
        algo = MCTSAlgorithm(n_iterations=20, sim_depth=10, seed=0)
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
# DQN helpers
# ---------------------------------------------------------------------------


class TestEncodeBoard:
    def test_empty_board_all_zeros(self):
        board = [[0] * 4 for _ in range(4)]
        enc = _encode_board(board)
        assert enc.shape == (16,)
        assert (enc == 0).all()

    def test_tile_2048_encodes_to_one(self):
        board = [[2048, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        enc = _encode_board(board)
        assert abs(enc[0] - 1.0) < 1e-6

    def test_tile_2_encodes_correctly(self):
        import math
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        enc = _encode_board(board)
        assert abs(enc[0] - math.log2(2) / 11.0) < 1e-6

    def test_length_is_16(self):
        board = [[2, 4, 8, 16], [32, 64, 128, 256],
                 [512, 1024, 2048, 4], [2, 2, 2, 2]]
        enc = _encode_board(board)
        assert len(enc) == 16


class TestBoardHeuristic:
    def test_empty_board_returns_zero(self):
        board = [[0] * 4 for _ in range(4)]
        assert _board_heuristic(board) == 0.0

    def test_higher_tiles_give_higher_score(self):
        low = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        high = [[1024, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert _board_heuristic(high) > _board_heuristic(low)


class TestQNetwork:
    def test_forward_output_shape_single(self):
        import numpy as np
        rng = np.random.default_rng(0)
        net = _QNetwork(16, 64, 4, rng)
        x = rng.random(16).astype("float32")
        q, *_ = net.forward(x)
        assert q.shape == (4,)

    def test_forward_output_shape_batch(self):
        import numpy as np
        rng = np.random.default_rng(0)
        net = _QNetwork(16, 64, 4, rng)
        x = rng.random((8, 16)).astype("float32")
        q, *_ = net.forward(x)
        assert q.shape == (8, 4)

    def test_copy_weights_produces_identical_outputs(self):
        import numpy as np
        rng = np.random.default_rng(1)
        net1 = _QNetwork(16, 64, 4, rng)
        net2 = _QNetwork(16, 64, 4, rng)
        net2.copy_weights(net1)
        x = rng.random(16).astype("float32")
        q1, *_ = net1.forward(x)
        q2, *_ = net2.forward(x)
        assert (q1 == q2).all()


# ---------------------------------------------------------------------------
# DQNAlgorithm
# ---------------------------------------------------------------------------


class TestDQNAlgorithm:
    def test_algorithm_name(self):
        assert DQNAlgorithm.name == "DQN"

    def test_choose_move_returns_valid_direction(self):
        algo = DQNAlgorithm(seed=0)
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_choose_move_restricted_to_valid_moves(self):
        """With epsilon=0 the chosen move must be valid (changes the board)."""
        algo = DQNAlgorithm(epsilon_start=0.0, epsilon_end=0.0, seed=0)
        board = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        for _ in range(10):
            move = algo.choose_move(board)
            new_board, _ = simulate_move(board, move)
            assert not _boards_equal(board, new_board), f"Chose invalid move {move!r}"

    def test_falls_back_on_fully_blocked_board(self):
        algo = DQNAlgorithm(seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_trains_after_enough_transitions(self):
        """Ensure no errors when the replay buffer fills and training is triggered."""
        algo = DQNAlgorithm(batch_size=4, buffer_size=20, seed=0)
        board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for _ in range(20):
            algo.choose_move(board)

    def test_epsilon_decays_over_time(self):
        algo = DQNAlgorithm(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=0.9, seed=0)
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        initial_epsilon = algo._epsilon
        for _ in range(10):
            algo.choose_move(board)
        assert algo._epsilon < initial_epsilon

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the DQN algorithm."""
        algo = DQNAlgorithm(seed=0)
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
# PPOAlgorithm
# ---------------------------------------------------------------------------


class TestPPOAlgorithm:
    def test_algorithm_name(self):
        assert PPOAlgorithm.name == "PPO"

    def test_choose_move_returns_valid_direction(self):
        algo = PPOAlgorithm(seed=0)
        board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_choose_move_restricted_to_valid_moves(self):
        """The chosen move must always be in DIRECTIONS."""
        algo = PPOAlgorithm(seed=0)
        board = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        for _ in range(10):
            move = algo.choose_move(board)
            assert move in DIRECTIONS

    def test_falls_back_on_fully_blocked_board(self):
        algo = PPOAlgorithm(seed=0)
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move = algo.choose_move(board)
        assert move in DIRECTIONS

    def test_ppo_update_triggered_after_update_freq_steps(self):
        """No errors should occur when the rollout buffer fills and PPO updates."""
        algo = PPOAlgorithm(update_freq=4, n_epochs=2, seed=0)
        board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for _ in range(10):
            algo.choose_move(board)

    def test_rollout_buffer_cleared_after_update(self):
        """After a PPO update the rollout buffer should be cleared."""
        algo = PPOAlgorithm(update_freq=4, seed=0)
        board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for _ in range(5):
            algo.choose_move(board)
        # Buffer should have been cleared after 4 steps; at most 1 entry remains.
        assert len(algo._buf_states) <= 1

    def test_full_game_runs_to_completion(self, game):
        """Play a full game with the PPO algorithm."""
        algo = PPOAlgorithm(seed=0)
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
