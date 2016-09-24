from collections import namedtuple
from games import (Game)
from pprint import pprint


# class GameState:
#     def __init__(self, to_move, board, label=None):
#         self.to_move = to_move
#         self.board = board
#         self.label = label
#
#     def __str__(self):
#         if self.label == None:
#             return super(GameState, self).__str__()
#         return self.label
#
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

# class DotsAndBoxes():
#
#     def __init__(self):
#         pass
#
#         pygame.init()
#         width, height = 240, 320
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Boxes")
#         self.boardh = [[False for x in range(3)] for y in range(4)]
#         self.boardv = [[False for x in range(4)] for y in range(3)]
#
#         self.initGraphics()
#
#     def Update(self):
#
#         self.screen.fill(0)
#         self.drawBoard()
#
#
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 exit()
#
#         pygame.display.flip()
#
# dg=DotsAndBoxes()
# while 1:
#     dg.Update()
#
#
# def initGraphics(self):
#         self.normallinev=pygame.image.load("normalline.png")
#         self.normallineh=pygame.transform.rotate(pygame.image.load("normalline.png"), -90)
#         self.bar_donev=pygame.image.load("bar_done.png")
#         self.bar_doneh=pygame.transform.rotate(pygame.image.load("bar_done.png"), -90)
#         self.hoverlinev=pygame.image.load("hoverline.png")
#         self.hoverlineh=pygame.transform.rotate(pygame.image.load("hoverline.png"), -90)
#
# def drawBoard(self):
#     for x in range(3):
#         for y in range (4):
#             if not self.boardh[y][x]:
#                 self.screen.blit(self.normallineh, [(x)*80, (y)*80])
#             else:
#                 self.screen.blit(self.bar_doneh, [(x)*80, (y)*80])
#     for x in range(4):
#         for y in range(3):
#             if not self.boardv[y][x]:
#                 self.screen.blit(self.normallinev, [(x)*80, (y)*80])
#             else:
#                 self.screen.blit(self.bar_donev, [(x)*80, (y)*80])
#
#
#
# from collections import namedtuple
# from games import (Game)
# import pygame





class GameState:
    def __init__(self, to_move, width, height, board, label=None):
        self.board = board
        self.width = width
        self.height = height
        self.to_move = to_move
        self.label = label

    # keep this mesthod... not sure its purpose

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label


class DotsAndBoxes(Game):
    """A flagrant copy of TicTacToe, from game.py
    It's simplified, so that moves and utility are calculated as needed
    Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    A copy of the game dots and boxes. This game is played on a board that is 4 by 4 square
    the goal is to create a completed square first.
    """

    def __init__(self, height=5, width=5):
        self.height = height
        self.width = width
        self.initial = GameState(to_move='----', width=3, height=3, board={})
        self.boardh = [[False for x in range(3)] for y in range(4)]
        self.boardv = [[False for x in range(4)] for y in range(3)]
        board = [['+'] * 5] * 5
        pprint(board)

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, self.height + 1):
            for y in range(1, self.width + 1):
                if (x, y) not in state.board.keys():
                    moves.append((x, y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == '----':
            return '-----'

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

    # begining of my methods

    def GameOver(self):
        # calculates if anymore moves can be made
        # returns true if no more moves can be made

        w, h = self.width, self.height
        return len(self.board.keys()) == 2 * w * h - h - w


def _isSquareMove(self, move):
    b = self.board
    mmove = self._makemove
    move = ((x1, y1), (x2, y2))
    captured_squares = []
    if self._isHorizontal(move):
        for j in [-1, 1]:
            if (b.has_key(mmove((x1, y1), (x1, y1 - j)))
                and b.has_key(mmove((x1, y1 - j), (x1 + 1, y1 - j)))
                and b.has_key(mmove((x1 + 1, y1 - j), (x2, y2)))):
                captured_squares.append(min([(x1, y1), (x1, y1 - j),
                                             (x1 + 1, y1 - j), (x2, y2)]))
    else:
        for j in [-1, 1]:
            if (b.has_key(mmove((x1, y1), (x1 - j, y1)))
                and b.has_key(mmove((x1 - j, y1), (x1 - j, y1 + 1)))
                and b.has_key(mmove((x1 - j, y1 + 1), (x2, y2)))):
                captured_squares.append(min([(x1, y1), (x1 - j, y1),
                                             (x1 - j, y1 + 1), (x2, y2)]))
    return captured_squares

def _isHorizontal(self, move):
    # return true is the move is horizontal
    return abs(move[0][0] - move[1][0]) == 1

def _idVertical(self, move):
    # return true if the move is vertical
    return not self._isHorizontal(self, move)





    # Did I win?
    # def check_win(self, board, player):
    #     # check rows
    #     for y in range(1, self.v + 1):
    #         if self.k_in_row(board, (1, y), player, (1, 0)):
    #             return 1
    #     # check columns
    #     for x in range(1, self.h + 1):
    #         if self.k_in_row(board, (x, 1), player, (0, 1)):
    #             return 1
    #     # check \ diagonal
    #     if self.k_in_row(board, (1, 1), player, (1, 1)):
    #         return 1
    #     # check / diagonal
    #     if self.k_in_row(board, (3, 1), player, (-1, 1)):
    #         return 1
    #     return 0
    #
    # # does player have K in a row? return 1 if so, 0 if not
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
    #
    # def terminal_test(self, state):
    #     "A state is terminal if it is won or there are no empty squares."
    #     return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0
    #
    # def display(self, state):
    #     board = state.board
    #     for x in range(1, self.h + 1):
    #         for y in range(1, self.v + 1):
    #             print(board.get((x, y), '.'), end=' ')
    #         print()


myGames = DotsAndBoxes()
