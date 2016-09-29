from collections import namedtuple
from games import (Game)

class GameState:
    def __init__(self, to_move, board, label=None):
        self.to_move = to_move
        self.board = board
        self.label = label
        # self.maxDepth = depth # depth = 15

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class dodgeEm(Game):
    """
       a 4x4 board and recreation of the board shown on the thinkfun website.
       the goal is for the player to have 4 pieces off the board (in a row/column)
        The square (5,1) is blank and is an
       invalid move for both Blue(B) and Yellow(Y).

       ~Whoever captures(1,5) - top right corner first wins
         - So, the best piece to move for Blue is the one that is in (1,1) and the best first
           piece for Yellow is (5,5)

              | YELLOW WINS |
               ______________ __
           (B)| •  •  •  •  | B
           (B)| •  •  •  •  | L
           (B)| •  •  •  •  | U
           (B)| •  •  •  •  | E
               –––––––––––––| WINS
           (X) (Y)(Y)(Y)(Y)   ––

"""
    def __init__(self, h=5, v=5):

        self.h = h
        self.v = v
        # self.k = k #k = 4
        self.initial = GameState(to_move='B', board={})
        self.validMoves = (1,2), (2,2), (3,2), (4,2)  # Assuming that B goes first
        self.gameOverYell = ((1,5), (2,5), (3,5), (4,5)) # Blue Wins
        self.gameOverBlue = ((1,2), (1,3), (1,4), (1,5)) # Yellow Wins


    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not taken and is to the left or right or up or down ."
        moves = []
        for x in range(1, self.h - 1):
            for y in range(1, self.v + 1):
                if (x,y) not in state.board.keys():
                    moves.append((x,y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'B':
            return 'Y'
        if player == 'B':
            return 'Y'
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
            util = -self.check_win(board, 'Y')
        state.utility = util
        return util if player == 'B' else -util

    # Did I win?
    def check_win(self, board, player):
        # check row - yellow wins
        for y in range(1, self.v + 1):
            if self.k_in_row(board, (1,y), player, (1,0)):
                return 1
        # check column - you win
        for x in range(1, self.h-1):
            if self.k_in_row(board, (x,5), player, (0,1)):
                return 1
        return 0

    # this compiles
    def check_win(self, board, player):
        # Did blue win?
        for (x,y) in self.gameOverYell[1:9]:
            if board.get((x,y)) == player:
                return 1
        for (x, y) in self.gameOverBlue[1:9]:
            if board.get((x,y)) == player:
                return 1
        return 0

    # def k_in_row(self, board, start, player, direction):
    #     "Return true if there is a line through start on board for player."
    #     (delta_x, delta_y) = direction
    #     x, y = start
    #     n = 0  # n is number of moves in row
    #     while board.get((x, y)) == player:
    #         n += 1
    #         x, y = x + delta_x, y + delta_y
    #     x, y = start
    #     while board.get((x, y)) == player:
    #         n += 1
    #         x, y = x - delta_x, y - delta_y
    #     n -= 1  # Because we counted start itself twice
    #     return n >= self.k

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'B') != 0 or len(self.actions(state)) == 0

    def display(self, state):

        board = state.board
        for x in range(2, self.h + 1):
            for y in range(1, self.v):
                print(board.get((x, y), '•'), end=' ')
            print()


myGame = dodgeEm()
#
blueWins = GameState(
    to_move = 'Y',
    board = {(1,5): 'B'

            },
    label = 'Blue Wins'
)

yellowWins = GameState(
    to_move = 'B',
    board = {(1,5): 'Y'

    },
    label='Yellow Wins'
)

# gameOverBlue = GameState(
#     to_move = 'Y',
#     board = {(1,3): 'B', (1,5): 'Y', (2,4): 'B',
#              (2,5): 'Y', (3,2): 'B', (3,5): 'Y',
#              (4,3): 'B',(4,5): 'Y',
#              },
#
#     label = 'Yellow wins'
# )


myGames = {
    myGame: [
       # gameOverBlue, gameOverYell
        blueWins, yellowWins
    ]
}