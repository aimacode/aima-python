from collections import namedtuple
from games import (Game)

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

class Gomoku(Game):
    """ The Japanese strategy based game of five in a row. It is orignally played on a 19x19 board,
    however, we don't have the memory for that. So it made a baby version of it that is 5x5"""

    def __init__(self, l=5, w=5, r=5):
        self.l = l
        self.w = w
        self.r = r
        self.initial = GameState(to_move='B', board={})

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, self.l + 1):
            for y in range(1, self.w + 1):
                if (x,y) not in state.board.keys():
                    moves.append((x,y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'B':
            return 'W'
        if player == 'W':
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
            util = -self.check_win(board, 'W')
        state.utility = util
        return util if player == 'B' else -util

    # Win?
    def check_win(self, board, player):
        # check rows
        for y in range(1, self.w + 1):
            if self.r_in_row(board, (1,y), player, (1,0)):
                return 1
        # check columns
        for x in range(1, self.l + 1):
            if self.r_in_row(board, (x,1), player, (0,1)):
                return 1
        # check \ diagonal
        if self.r_in_row(board, (5,1), player, (1,1)):
            return 1
        # check / diagonal
        if self.r_in_row(board, (5,1), player, (-1,1)):
            return 1
        return 0

    # player has a winning solution?
    def r_in_row(self, board, start, player, direction):
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
        return n >= self.r

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'B') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.l + 1):
            for y in range(1, self.w + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

myGame = Gomoku()

won = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'B', (1,3): 'B', (1,4): 'B', (1,5): 'B',
             (2,1): 'W', (2,2): 'W', (2,3): 'W', (2,4): 'W'
            },
    label = 'won'
)
won2 = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (2,1): 'B', (3,1): 'B', (4,1): 'B', (5,1): 'B',
             (2,4): 'W', (2,2): 'W', (2,3): 'W', (2,4): 'W'
            },
    label = 'won'
)

lost = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'B', (1,3): 'B', (1,4): 'B',
             (2,1): 'W', (2,2): 'W', (2,3): 'W', (2,4): 'W', (2,5): 'W',
             (3,1): 'B'
            },
    label = 'lost'
)

win1 = GameState(
    to_move = 'B',
    board = {(3,1): 'B', (3,2): 'B', (3,3): 'B', (3,4): 'B',
             (2,1): 'W', (2,2): 'W', (2,3): 'W',
            },
    label = 'win by 1'
)
myGames = {
    myGame: [
        won,
        lost,
       # win1,
        won2
    ],

}