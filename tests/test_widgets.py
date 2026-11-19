import random

import pytest

from aima.csp import MapColoringCSP, NQueensCSP, AC3, backtracking_search, forward_checking, min_conflicts
from aima.games import Fig52Game, alpha_beta_search, minmax_decision
from aima.planning import GraphPlan, linearize, spare_tire, three_block_tower
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


def test_game_tree_player():
    player = game_tree_player(alpha_beta_search, Fig52Game())
    assert len(player.frames) == 11
    player.slider.value = 6  # leaf C1, after which C is done
    assert 'back up C = 2' in player.view.value and 'pruned' in player.view.value
    player.slider.value = player.slider.max
    assert 'best move is <b>a1</b>, with value 3' in player.view.value
    assert '2 actions pruned' in player.view.value
    player = game_tree_player(minmax_decision, Fig52Game())
    player.slider.value = player.slider.max
    assert '0 actions pruned' in player.view.value


def test_tic_tac_toe_game():
    game = TicTacToeGame()
    game.click((1, 1))
    assert game.state.board[(1, 1)] == 'X' and len(game.state.board) == 2
    assert 'alpha-beta looked at' in game.message.value
    game.click((1, 1))
    assert 'taken' in game.message.value and len(game.state.board) == 2
    random.seed(1)
    game.opponent.value = 'random'
    while not game.game.terminal_test(game.state):
        game.click(game.state.moves[0])
    assert any(word in game.message.value for word in ('you win', 'O wins', 'draw'))


def test_plan_player():
    problem = three_block_tower()
    plan = linearize(GraphPlan(three_block_tower()).execute())
    player = plan_player(problem, plan)
    assert len(player.frames) == len(plan) + 1 and 'Goals holding: 0 of 2' in player.view.value
    player.slider.value = player.slider.max
    assert 'Goals holding: 2 of 2' in player.view.value
    assert not problem.goal_test()  # the player works on a copy
    player = plan_player(spare_tire(), linearize(GraphPlan(spare_tire()).execute()))
    player.step()
    assert 'line-through' in player.view.value


def test_graph_plan_player():
    player = graph_plan_player(spare_tire())
    assert 'S0' in player.view.value and 'A0' not in player.view.value
    player.slider.value = player.slider.max
    assert 'PutOn(Spare, Axle)' in player.view.value and 'actions of the plan' in player.view.value
    plan = GraphPlan(spare_tire()).execute()[0]
    green = player.view.value.count('color:#2a9d8f;font-weight:bold')
    assert green == sum(1 for step in plan for action in step if action.op[0] != 'P' or not action.op[1].isupper())


if __name__ == '__main__':
    pytest.main()
