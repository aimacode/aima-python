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

class Hex(Game):
    """ """

<<<<<<< HEAD
    def __init__(self, h=5, v=5):
        self.h = h
        self.v = v
        # self.blueWin = ((1, 1), (1, 2), (1, 3))
        # self.redWin = ((1, 3), (2, 3), (3, 3))
=======
    def __init__(self, h=3, v=3):
        self.h = h
        self.v = v
        self.blueWin = ((1, 1), (1, 2), (1, 3))
        self.redWin = ((1, 3), (2, 3), (3, 3))
>>>>>>> 0a0482d1b91e5182f5bf33639406810aa781fa6e
        self.initial = GameState(to_move='B', board={})

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                if (x, y) not in state.board.keys():
                    moves.append((x, y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'B':
            return 'R'
        if player == 'R':
            return 'B'
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
            return state.utility if player == 'B' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, 'B')
        if util == 0:
            util = -self.check_win(board, 'R')
        state.utility = util
        return util if player == 'B' else -util

    # Did I win?
    def check_win(self, board, player):
        if player == 'B':
            for y in range(1, self.v + 1):
                if board.get((1, y)) == player and self.check_blue(board, (1, y), player, (1, 0)) == 1:
                    return 1
            return 0
        if player == 'R':
            for x in range(1, self.h + 1):
<<<<<<< HEAD
                if board.get((x, 1)) == player and self.check_red(board, (x, 1), player, (0, 1)) == 1:
=======
                if board.get((x, 3)) == player and self.check_red(board, (x, 3), player, (0, -1)) == 1:
>>>>>>> 0a0482d1b91e5182f5bf33639406810aa781fa6e
                    return 1
            return 0
        else:
            return 0

    def check_blue(self, board, start, player, direction):
        (delta_x, delta_y) = direction
        (start_x, start_y) = start
        x, y = start_x + delta_x, start_y + delta_y
        # check
        if board.get((x, y)) == player:
<<<<<<< HEAD
            if x == self.h:
                return 1
            else:
                return self.check_blue(board, (x, y), player, (1, 0))
=======
            return 1
>>>>>>> 0a0482d1b91e5182f5bf33639406810aa781fa6e
        else:
            if direction == (1, 0):     # if down
                direction = (1, -1)     # down and left
            elif direction == (1, -1):  # if down and left
                direction = (1, 1)      # down and right
            else:
                return 0
            return self.check_blue(board, start, player, direction)

    def check_red(self, board, start, player, direction):
        (delta_x, delta_y) = direction
        (start_x, start_y) = start
        x, y = start_x + delta_x, start_y + delta_y
        # check
        if board.get((x, y)) == player:
<<<<<<< HEAD
            if y == self.v:
                return 1
            else:
                return self.check_red(board, (x, y), player, (0, 1))
        else:
            if direction == (0, 1):    # if right
                direction = (-1, 1)    # right and up
            elif direction == (-1, 1): # if right and up
                direction = (1, 1)     # right and down
=======
            return 1
        else:
            if direction == (0, -1):    # if left
                direction = (-1, -1)    # left and up
            elif direction == (-1, -1): # if left and up
                direction = (1, -1)     # left and down
>>>>>>> 0a0482d1b91e5182f5bf33639406810aa781fa6e
            else:
                return 0
            return self.check_red(board, start, player, direction)

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'B') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = Hex()



myGames = {
    myGame: [

    ]
}