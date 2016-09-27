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

# class FlagrantCopy(Game):
#     """A flagrant copy of TicTacToe, from game.py
#     It's simplified, so that moves and utility are calculated as needed
#     Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
#     A state has the player to move and a board, in the form of
#     a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""
#
#     def __init__(self, h=3, v=3, k=3):
#         self.h = h
#         self.v = v
#         self.k = k
#         self.initial = GameState(to_move='X', board={})
#
#     def actions(self, state):
#         try:
#             return state.moves
#         except:
#             pass
#         "Legal moves are any square not yet taken."
#         moves = []
#         for x in range(1, self.h + 1):
#             for y in range(1, self.v + 1):
#                 if (x,y) not in state.board.keys():
#                     moves.append((x,y))
#         state.moves = moves
#         return moves
#
#     # defines the order of play
#     def opponent(self, player):
#         if player == 'X':
#             return 'O'
#         if player == 'O':
#             return 'X'
#         return None
#
#     def result(self, state, move):
#         if move not in self.actions(state):
#             return state  # Illegal move has no effect
#         board = state.board.copy()
#         player = state.to_move
#         board[move] = player
#         next_mover = self.opponent(player)
#         return GameState(to_move=next_mover, board=board)
#
#     def utility(self, state, player):
#         "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
#         try:
#             return state.utility if player == 'X' else -state.utility
#         except:
#             pass
#         board = state.board
#         util = self.check_win(board, 'X')
#         if util == 0:
#             util = -self.check_win(board, 'O')
#         state.utility = util
#         return util if player == 'X' else -util
#
#     # Did I win?
#     def check_win(self, board, player):
#         # check rows
#         for y in range(1, self.v + 1):
#             if self.k_in_row(board, (1,y), player, (1,0)):
#                 return 1
#         # check columns
#         for x in range(1, self.h + 1):
#             if self.k_in_row(board, (x,1), player, (0,1)):
#                 return 1
#         # check \ diagonal
#         if self.k_in_row(board, (1,1), player, (1,1)):
#             return 1
#         # check / diagonal
#         if self.k_in_row(board, (3,1), player, (-1,1)):
#             return 1
#         return 0
#
#     # does player have K in a row? return 1 if so, 0 if not
#     def k_in_row(self, board, start, player, direction):
#         "Return true if there is a line through start on board for player."
#         (delta_x, delta_y) = direction
#         x, y = start
#         n = 0  # n is number of moves in row
#         while board.get((x, y)) == player:
#             n += 1
#             x, y = x + delta_x, y + delta_y
#         x, y = start
#         while board.get((x, y)) == player:
#             n += 1
#             x, y = x - delta_x, y - delta_y
#         n -= 1  # Because we counted start itself twice
#         return n >= self.k
#
#     def terminal_test(self, state):
#         "A state is terminal if it is won or there are no empty squares."
#         return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0
#
#     def display(self, state):
#         board = state.board
#         for x in range(1, self.h + 1):
#             for y in range(1, self.v + 1):
#                 print(board.get((x, y), '.'), end=' ')
#             print()
#
#
# myGame = FlagrantCopy()
#
# won = GameState(
#     to_move = 'O',
#     board = {(1,1): 'X', (1,2): 'X', (1,3): 'X',
#              (2,1): 'O', (2,2): 'O',
#             },
#     label = 'won'
# )
#
# winin1 = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'X',
#              (2,1): 'O', (2,2): 'O',
#             },
#     label = 'winin1'
# )
#
# losein1 = GameState(
#     to_move = 'O',
#     board = {(1,1): 'X', (1,2): 'X',
#              (2,1): 'O', (2,2): 'O',
#              (3,1): 'X',
#             },
#     label = 'losein1'
# )
#
# winin3 = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'O',
#              (2,1): 'X',
#              (3,1): 'O',
#             },
#     label = 'winin3'
# )
#
# losein3 = GameState(
#     to_move = 'O',
#     board = {(1,1): 'X',
#              (2,1): 'X',
#              (3,1): 'O', (1,2): 'X', (1,2): 'O',
#             },
#     label = 'losein3'
# )
#
# winin5 = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'O',
#              (2,1): 'X',
#             },
#     label = 'winin5'
# )
#
# lost = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'X',
#              (2,1): 'O', (2,2): 'O', (2,3): 'O',
#              (3,1): 'X'
#             },
#     label = 'lost'
# )
#
# myGames = {
#     myGame: [
#         won,
#         winin1, losein1, winin3, losein3, winin5,
#         lost,
#     ]
# }

class CTT(Game):
    """A version of the Circle-Tac-Toe game
    It's simplified, so that moves and utility are calculated as needed
    Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=7, v=7, k=3, depth=2):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={})
        self.maxDepth = depth
        self.invalidSpaces = {
            (1,2):'H', (1,3):'H', (1,4):'H', (1,5):'H', (1,6):'H', (2,1):'H', (2,3):'H',
            (2, 4):'H', (2,5):'H', (2,7):'H', (3,1):'H', (3,2):'H', (3,4):'H', (3,6):'H',
            (3, 7):'H', (4,1):'H', (4,2):'H', (4,3):'H', (4,4):'H',(4,5):'H',(4,6):'H',(4,7):'H',
            (5, 1):'H',(5,2):'H',(5,4):'H',(5,6):'H',(5,7):'H',(6,1):'H',(6,3):'H',(6,4):'H',(6,5):'H',
            (6, 7):'H',(7,2):'H',(7,3):'H',(7,4):'H',(7,5):'H',(7,6):'H'
        }
        #Valid spaces make up the spaces within the 7x7 grid that are "allowable".
        #This is set as an immutable Tuple since it will be true always and can't be changed.
        self.validSpaces = (
        (1, 1), (1, 7), (2, 2), (2, 6), (3, 3), (3, 5), (5, 4), (6, 4),(7,4),)
        #c1, c2, c3, d1, d2, and d3 are the coordinates that make up the 6 winning states.
        #these will be checked when determining if the game is over.
        self.c1 = ((3,3), (3,5), (5,4),)
        self.c2 = ((2,2), (2,6), (6,4),)
        self.c3 = ((1,1), (1,7), (7,4),)
        self.d1 = ((1,1), (2,2), (3,3),)
        self.d2 = ((1,7), (2,6), (3,5),)
        self.d3 = ((5,4), (6,4), (7,4),)



    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any 'valid' squares not yet taken."
        moves = []
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                if (x,y) in self.validSpaces:
                    if (x,y) not in state.board.keys():
                        moves.append((x,y))
                # if (x,y) not in state.board.keys():
                #     moves.append((x,y))
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

        # check d1
        if self.k_in_row(self.d1,board,player) == 3:
         return 1
        # check d2
        if self.k_in_row(self.d2,board,player) == 3:
         return 1
        # check d3
        if self.k_in_row(self.d3,board,player) == 3:
         return 1
        # check c1
        if self.k_in_row(self.c1,board,player) == 3:
         return 1
        # check c2
        if self.k_in_row(self.c2,board,player) == 3:
         return 1
        # check c3
        if self.k_in_row(self.c3,board,player) == 3:
         return 1
        return 0

    # does player have 3 in a row? return 1 if so, 0 if not
    def k_in_row(self,list,board,player):
        x=0
        y=0
        while x<3:
            coordinate = list[x]
            if board.get (coordinate) == player:
                y += 1
            x += 1
        return y


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
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = CTT()

won = GameState(
    to_move = 'O',
    board = {(1,1): 'X', (2,2): 'X', (3,3): 'X',
             (1,7): 'O', (2,6): 'O',
            },
    label = 'won'
)

winin1 = GameState(
    to_move = 'X',
    board = {(3,3): 'X', (3,3): 'X',
             (6,4): 'O', (7,4): 'O',
            },
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'O',
    board={(3, 3): 'X', (3, 3): 'X', (5,4): 'X',
           (6, 4): 'O', (7, 4): 'O',
           },
    label = 'losein1'
)


winin2 = GameState(
    to_move = 'X',
    board = {(1,1): 'O', (1,7): 'X', (2,6): 'O',
             (3,3): 'X', (6,4): 'X', (7,4): 'O'
            },
    label = 'winin2'
)


#
stalemate = GameState(
    to_move = 'X',
    board = {(1,1): 'X', (1,7): 'O',
             (2,2): 'O', (2,6): 'X',
             (3,3): 'X',(3,5): 'O',(5,4): 'X',
             (6,4): 'O',(7,4): 'X',
            },
    label = 'stalemate'
)


new = GameState(
    to_move = 'X',
    board = { },
    label = 'new'
)
#
# losein3 = GameState(
#     to_move = 'O',
#     board = {(1,1): 'X',
#              (2,1): 'X',
#              (3,1): 'O', (1,2): 'X', (1,2): 'O',
#             },
#     label = 'losein3'
# )
#
# winin5 = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'O',
#              (2,1): 'X',
#             },
#     label = 'winin5'
# )

# lost = GameState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'X',
#              (2,1): 'O', (2,2): 'O', (2,3): 'O',
#              (3,1): 'X'
#             },
#     label = 'lost'


myGames = {
    myGame: [
        won,
        winin1, losein1,
        winin2, stalemate, new
    ]
}

