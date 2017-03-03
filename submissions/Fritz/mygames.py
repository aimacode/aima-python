from collections import namedtuple
from games import (Game)

class GameState:
    def __init__(self, to_move, board, label=None, maxDepth=5):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = maxDepth

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Connect4(Game):
    """
    A game of connect 4 played on a 7x7 board.
    """

    def __init__(self, h=7, v=7, k=4):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={})
        ##this is a tests

    def land(self, board, column):
        # determines where the current top game piece is in a column
        for row in reversed(range(self.v)):
            position = (row, column)
            if position in board.keys():
                continue
            return position

    def hasRoom(self, board, column):
        # checks if a column is full
        topRow = self.v - 7
        topCell = (topRow, column)
        if topCell in board.keys():
            return False
        else:
            return True

    def actions(self, state):
        # defines the available moves at any point in the game
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(self.h):
            if self.hasRoom(state.board, x):
                moves.append(x)
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
        position = self.land(board, move)
        board[position] = player
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
        # check columns
        for y in range(self.v):
            if self.checkColumnWin(board, y, player) == self.k:
            #if self.k_in_row(board, (1,y), player, (1,0)):
                return -1
        # check rows
        for x in range(self.h):
            if self.checkRowWin(board, x, player) == self.k:
                return -1
        # check / diagonal
        for y in range(self.v):
            if self.checkDiagonalWin(board, (6,y), player, (-1,1)) == self.k:
                return -1
        # check \ diagonal
        for y in range(self.v):
            if self.checkDiagonalWin(board, (6,y), player, (-1,-1)):
                return -1
        return 0

    def checkRowWin(self, board, row, player):
        # checks if there are 4 connected in any row
        count = 0
        for column in range(self.v):
            position = (row, column)
            if board.get(position) == player:
                count = count +1
                if count == self.k:
                    return count
            else: count = 0

    def checkColumnWin(self, board, column, player):
        #checks if there are 4 connected in any column
        count = 0
        for row in range(self.h):
            position = (row, column)
            if board.get(position) == player:
                count = count +1
                if count == self.k:
                    return count
            else: count = 0

    def checkDiagonalWin(self, board, start, player, direction):
        #checks if there are 4 connected on any diagonal
        count = 0
        (d_x, d_y) = direction
        x, y = start
        while board.get((x, y)) == player:
            count = count + 1
            x = x + d_x
            y = y + d_y
            if count == self.k:
                return count

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        # prints out a visual display of the current status of the board
        board = state.board
        for x in range(self.h):
            for y in range(self.v):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = Connect4()

won = GameState(
    to_move = 'O',
    board = {
            (5,1): 'O', (5,2): 'O', (5,3): 'O',
            (6,1): 'X', (6,2): 'X', (6,3): 'X', (6,4): 'X'
            },
    label = 'won'
)

winin1 = GameState(
     to_move = 'X',
     board = {
             (5,1): 'O', (5,2): 'O', (5,3): 'O',
             (6,1): 'X', (6,2): 'X', (6,3): 'X'
             },
     label = 'winin1'
)

losein1 = GameState(
    to_move = 'O',
    board = {
            (5,1): 'X', (5,2): 'X',             (5,4): 'X',
            (6,1): 'X', (6,2): 'O', (6,3): 'O', (6,4): 'O'
            },
    label = 'losein1'
)

losein2 = GameState(
    to_move = 'O',
    board = {
            (5,1): 'O', (5,2): 'X',             (5,4): 'X',
            (6,1): 'O', (6,2): 'X', (6,3): 'O', (6,4): 'X'
            },
    label = 'losein2'
)

winin2 = GameState(
    to_move = 'X',
    board = {
            (5,1): 'O',             (5,3): 'O',
            (6,1): 'X',             (6,3): 'X'
            },
    label = 'winin2'
)
lost = GameState(
    to_move = 'X',
    board = {           (0,2): 'O', (0,3): 'X',
            (1,1): 'X', (1,2): 'O', (1,3): 'O',
            (2,1): 'X', (2,2): 'O', (2,3): 'X',
            (3,1): 'X', (3,2): 'O', (3,3): 'O',
            (4,1): 'O', (4,2): 'X', (4,3): 'X',
            (5,1): 'O', (5,2): 'O', (5,3): 'O', (5,2): 'X', (5,3): 'X',
            (6,1): 'X', (6,2): 'X', (6,3): 'X', (6,2): 'O', (6,3): 'O'
            },
    label = 'lost'
)

myGames = {
    myGame: [
        won, winin1, losein1, winin2, lost, losein2
    ]
}