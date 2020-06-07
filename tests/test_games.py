import pytest

from games import *

# Creating the game instances
f52 = Fig52Game()
ttt = TicTacToe()

random.seed("aima-python")


def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


def test_minmax_decision():
    assert minmax_decision('A', f52) == 'a1'
    assert minmax_decision('B', f52) == 'b1'
    assert minmax_decision('C', f52) == 'c1'
    assert minmax_decision('D', f52) == 'd3'


def test_alpha_beta_search():
    assert alpha_beta_search('A', f52) == 'a1'
    assert alpha_beta_search('B', f52) == 'b1'
    assert alpha_beta_search('C', f52) == 'c1'
    assert alpha_beta_search('D', f52) == 'd3'

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alpha_beta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1), (3, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alpha_beta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1)],
                      o_positions=[])
    assert alpha_beta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 1)],
                      o_positions=[(2, 2), (3, 1)])
    assert alpha_beta_search(state, ttt) == (1, 3)


def test_random_tests():
    assert Fig52Game().play_game(alpha_beta_player, alpha_beta_player) == 3

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert ttt.play_game(alpha_beta_player, alpha_beta_player) >= 0

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert ttt.play_game(alpha_beta_player, random_player) >= 0


if __name__ == "__main__":
    pytest.main()
