from games import (Game)
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy

class GameState:
    def __init__(self, to_move, board, label=None, depth=8):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = depth

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label



class ConnectFour(Game):
    """
    An Alpha Beta implementation of Connect Four.
    """
    def __init__(self, h=6, v=7, k=4):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={})

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        moves = []
        for x in range(1, self.v + 1):
            for y in range(self.h, 0, -1):
                if (y, x) not in state.board.keys():
                    moves.append((y, x))
                    break
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
        player = state.to_move
        board[move] = player
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, board=board)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        try:
            return state.utility if player == 'X' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, 'X')
        if util == 0:
            util = -self.check_win(board, 'O')
        state.utility = util
        return util if player == 'X' else -util

    def check_win(self, board, player):
        # self.v = 7, self.h = 6
        # check columns
        for y in range(1, self.h + 1):
            for x in range(self.v - 1, 3, -1):
                if self.k_in_row(board, (x, y), player, (-1, 0)):
                    return 1
        # check rows
        for x in range(self.h, 0, -1):
            for y in range(1, self.v - 1):
                if self.k_in_row(board, (x, y), player, (0, 1)):
                    return 1

        # \ Win Check
        for y in range(self.v, 3, -1):
            for x in range(self.h, 2, -1):
                if self.k_in_row(board, (x, y), player, (-1, -1)):
                    return 1

        # / Win Check
        for y in range(1, self.h - 1):
            for x in range(self.v - 1, 2, -1):
                if self.k_in_row(board, (x, y), player, (-1, 1)):
                    return 1
        return 0

    def k_in_row(self, board, start, player, direction):
        "Return true if there is a line through start on board for player."
        (delta_x, delta_y) = direction
        x, y = start
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = start
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted start itself twice
        return n >= self.k

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = ConnectFour()

empty = GameState(
    to_move='0',
    board={},
    label='empty'
)

default = GameState(
    to_move='O',
    board={(6, 1): 'X', (6, 4): 'O'},
    label='default'
)

oneToWin = GameState(
    to_move='X',
    board={(6, 4): 'X', (6, 5): 'X', (6, 6): 'X', (6, 3): 'O', (6, 2): 'O', (6, 1): 'O'},
    label='one to win X'
)

myGames = {
    myGame: [default, oneToWin]
}