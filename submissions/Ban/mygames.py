from collections import namedtuple
from games import (Game)

class GameState:
    def __init__(self, to_move, board, label=None):
        self.to_move = to_move
        self.board = board
        self.label = label

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Nim(Game):
    """A flagrant copy of Nim, from game.py
    It's simplified, so that moves and utility are calculated as needed"""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board=[3, 3, 3])

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, state.board[0] + 1):
            moves.append(('a',x))
        for x in range(1, state.board[1] + 1):
            moves.append(('b',x))
        for x in range(1, state.board[2] + 1):
            moves.append(('c',x))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'X':
            return 'O'
        if player == 'O':
            return 'X'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = state.board.copy()
        if move[0] == 'a':
            board[0] -= move[1]
        if move[0] == 'b':
            board[1] -= move[1]
        if move[0] == 'c':
            board[2] -= move[1]
        player = state.to_move
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, board=board)

    def utility(self, state, player):
        board = state.board
        util = self.check_win(board)
        state.utility = util
        lastPlayer = state.to_move
        return util if lastPlayer == 'X' else -util

    # Did I win?
    def check_win(self, board):
        if board[0] == 0 and board[1] == 0 and board[2] == 0:
            return 1
        return 0


    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        for x in range(1, state.board[0] + 1):
            print('+'),
        print()
        for x in range(1, state.board[1] + 1):
            print('+'),
        print()
        for x in range(1, state.board[2] + 1):
            print('+'),
        print()


myGame = Nim()

none = GameState(
    to_move = 'O',
    board = [0,0,0],
    label = 'won'
)

won = GameState(
    to_move = 'O',
    board = [1,0,0],
    label = 'won'
)

winin1 = GameState(
    to_move = 'X',
    board = [2,1,0],
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'O',
    board = [1,1,0],
    label = 'losein1'
)

winin3 = GameState(
    to_move = 'X',
    board = [1,2,1],
    label = 'winin3'
)

losein3 = GameState(
    to_move = 'O',
    board = [1,2,1],
    label = 'losein3'
)

winin5 = GameState(
    to_move = 'X',
    board = [3,3,3],
    label = 'winin5'
)

lost = GameState(
    to_move = 'O',
    board = [1,2,3],
    label = 'lost'
)

myGames = {
    myGame: [
        none,
        won,
        winin1,
        losein1,
        winin3,
        losein3,
        winin5,
        lost,
    ]
}