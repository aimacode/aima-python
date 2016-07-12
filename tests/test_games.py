"""A lightweight test suite for games.py"""

# You can run this test suite by doing: py.test tests/games.py
# Of course you need to have py.test installed to do this.

import pytest

from games import *  # noqa

# Creating the game instances
f52 = Fig52Game()
ttt = TicTacToe()


def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3, k=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) \
        - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


def test_minimax_decision():
    assert minimax_decision('A', f52) == 'a1'
    assert minimax_decision('B', f52) == 'b1'
    assert minimax_decision('C', f52) == 'c1'
    assert minimax_decision('D', f52) == 'd3'


def test_alphabeta_full_search():
    assert alphabeta_full_search('A', f52) == 'a1'
    assert alphabeta_full_search('B', f52) == 'b1'
    assert alphabeta_full_search('C', f52) == 'c1'
    assert alphabeta_full_search('D', f52) == 'd3'

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alphabeta_full_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1), (3, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alphabeta_full_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1)],
                      o_positions=[])
    assert alphabeta_full_search(state, ttt) == (2, 2)

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 1)],
                      o_positions=[(2, 2), (3, 1)])
    assert alphabeta_full_search(state, ttt) == (1, 3)


def test_random_tests():
    assert play_game(Fig52Game(), alphabeta_player, alphabeta_player) == 3

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert play_game(ttt, alphabeta_player, alphabeta_player) >= 0

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert play_game(ttt, alphabeta_player, random_player) >= 0


if __name__ == '__main__':
    pytest.main()
