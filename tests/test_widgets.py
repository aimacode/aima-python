import random

import pytest

from aima.csp import MapColoringCSP, NQueensCSP, AC3, backtracking_search, forward_checking, min_conflicts
from aima.search import *
from aima.widgets import *


def test_graph_search_player():
    player = graph_search_player(astar_search, GraphProblem('Arad', 'Bucharest', romania_map))
    assert player.value == 0 and 'Step 0 of' in player.view.value
    player.step()
    assert player.value == 1 and 'expand <b>Arad</b>' in player.view.value
    player.slider.value = player.slider.max
    assert 'Goal reached' in player.view.value and '(cost 418)' in player.view.value
    player.reset()
    assert player.value == 0 and not player.play_button.value
    player.speed.value = 2
    assert player.timer.interval == 500


def test_eight_puzzle_player():
    player = eight_puzzle_player(astar_search, EightPuzzle((1, 2, 3, 4, 0, 6, 7, 5, 8)))
    assert player.frames == [(1, 2, 3, 4, 0, 6, 7, 5, 8), (1, 2, 3, 4, 5, 6, 7, 0, 8),
                             (1, 2, 3, 4, 5, 6, 7, 8, 0)]
    assert '@keyframes' not in player.view.value
    player.step()  # one move: the tile slides
    assert '@keyframes' in player.view.value
    player.slider.value = 0
    player.slider.value = 2  # a jump: no slide
    assert '@keyframes' not in player.view.value


def test_n_queens_player():
    player = n_queens_player(depth_first_tree_search, NQueensProblem(4))
    player.slider.value = player.slider.max
    assert 'Solution found' in player.view.value
    assert n_queens_svg((1, 3, 0, 2)).count('#d9534f') == 0  # a solution: no queen attacked
    assert n_queens_svg({0: 0, 1: 1}).count('#d9534f') == 2  # same diagonal


def test_eight_puzzle_game():
    assert scrambled_eight_puzzle(30, seed=1) == scrambled_eight_puzzle(30, seed=1)
    assert EightPuzzle(scrambled_eight_puzzle(30, seed=2)).check_solvability(scrambled_eight_puzzle(30, seed=2))
    game = EightPuzzleGame(initial=(1, 2, 3, 4, 5, 6, 0, 7, 8))
    assert 'aima-movable' in game.buttons[7]._dom_classes and 'aima-movable' not in game.buttons[0]._dom_classes
    game.click(0)
    assert game.moves == 0 and 'cannot move' in game.message.value
    game.click(7)
    game.click(8)
    assert game.solved and game.moves == 2
    assert 'Moves: 2' in game.message.value and 'yours is optimal' in game.message.value
    game.start((1, 2, 3, 4, 5, 6, 7, 0, 8))
    game.click(6)  # a legal but useless move, then back and forward
    game.click(7)
    game.click(8)
    assert game.solved and '2 moves longer than the optimum' in game.message.value


def australia():
    return MapColoringCSP(list('RGB'), 'SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ')


def test_csp_player():
    player = csp_player(backtracking_search, australia(), inference=forward_checking)
    assert 'Start' in player.view.value
    player.step()
    assert 'assign <b>SA = R</b>' in player.view.value
    player.slider.value = player.slider.max
    assert player.frames[-1]['assignment'] == backtracking_search(australia(), inference=forward_checking)
    assert 'Solved' in player.view.value
    csp = australia()
    csp.support_pruning()
    player = csp_player(lambda c: AC3(c, removals=c.suppose('WA', 'R') + c.suppose('Q', 'G')), csp)
    player.slider.value = player.slider.max
    assert [] in player.frames[-1]['domains'].values() and 'Inconsistent' in player.view.value
    random.seed(0)
    player = csp_player(min_conflicts, NQueensCSP(4), max_steps=100)
    player.slider.value = player.slider.max
    assert player.view.value.count('Q</text>') == 4 and 'Solved' in player.view.value


def test_map_coloring_game():
    game = MapColoringGame(australia())
    game.click('SA')
    game.click('WA')  # same color as SA
    assert 'WA' not in game.assignment and 'cannot be R' in game.message.value
    for value, regions in (('G', ['WA', 'Q', 'V', 'T']), ('B', ['NT', 'NSW'])):
        game.color.value = value
        for region in regions:
            game.click(region)
    assert 'solved' in game.message.value


if __name__ == '__main__':
    pytest.main()
