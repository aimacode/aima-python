from collections import namedtuple
from games import (Game)
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

class DucksinRow(Game):
    """A flagrant copy of TicTacToe, from game.py
    It's simplified, so that moves and utility are calculated as needed
    Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=5, v=5, k=4):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={(1,1): 'X', (1,2): 'O', (1,3): 'X', (1,4): 'O', (1,5): 'X',
            (3,1): 'X', (3,5): 'O',
            (5,1): 'O', (5,2): 'X', (5,3): 'O', (5,4): 'X', (5,5): 'O'})

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                if (x,y) not in state.board.keys():
                    moves.append((x,y))
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

    # Did I win?
    def check_win(self, board, player):
        # check rows
        for x in range(1, self.h):
            for y in range(1, self.v):
                if self.k_in_row(board, (x,y), player, (1,0)):
                    return 1

        # check columns
        for y in range(1, self.v):
            for x in range(1, self.h):
                if self.k_in_row(board, (x,y), player, (0,1)):
                    return 1

        # check \ diagonal
        if self.k_in_row(board, (1,1), player, (1,1)):
            return 1
        elif self.k_in_row(board, (2,1), player, (1,1)):
            return 1
        elif self.k_in_row(board, (1,2), player, (1,1)):
            return 1

        # check / diagonal
        if self.k_in_row(board, (1,5), player, (-1,1)):
            return 1
        elif self.k_in_row(board, (2,5), player, (-1,1)):
            return 1
        elif self.k_in_row(board, (1,4), player, (-1,1)):
            return 1
        return 0

    # does player have K in a row? return 1 if so, 0 if not
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


myGame = DucksinRow()


#
# start = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'O', (1,3): 'X', (1,4): 'O', (1,5): 'X',
#             (3,1): 'X', (3,5): 'O',
#             (5,1): 'O', (5,2): 'X', (5,3): 'O', (5,4): 'X', (5,5): 'O'
#             },
#     label = 'start'
# )

won = GameState(
    to_move = 'O',
    board = {(1,1): 'X', (1,2): 'O', (1,3): 'X', (1,4): 'O', (1,5): 'X',
             (2,4): 'X', (2,5): 'O',
             (3,1): 'X', (3,3): 'X', (3,4): 'O',  (3,5): 'O',
             (4,2): 'X',
             (5,1): 'O', (5,2): 'X', (5,3): 'O', (5,4): 'X', (5,5): 'O'
            },
    label = 'won'
)


winin1 = GameState(
    to_move='X',
    board={
           (3, 3): 'X',
           (4, 1): 'O', (4, 2): 'X', (4, 3): 'O',
           (5, 1): 'X', (5, 3): 'O',
           },
    label = 'winin1'
)

losein1 = GameState(
    to_move='X',
    board={
           (2, 2): 'O', (2,3): 'O', (2,4): 'O', (2,5): 'X',
           (4, 3): 'X', (4, 4): 'X',
           (5, 3): 'X', (5, 4): 'O',
           },
    label = 'losein1'
)

lost = GameState(
    to_move = 'X',
    board = {(1,1): 'X', (1,2): 'O', (1,3): 'X', (1,4): 'O', (1,5): 'X',
             (2,1): 'O', (2,2): 'O', (2,3): 'O', (2,4): 'X', (2,5): 'X',
             (3,1): 'X', (3,2): 'X', (3,3): 'O', (3,4): 'X', (3,5): 'O',
             (4,1):'X', (4,2): 'X', (4,3): 'O', (4,4): 'O',
             (5,1): 'O', (5,2):'X', (5,3): 'O', (5,4): 'X', (5,5): 'O',
            },
    label = 'lost'
)



myGames = {
    myGame: [
        won,
        # winin1,
        # losein1,
        lost,
    ],

}
