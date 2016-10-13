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

class FlagrantCopy(Game):
    """A flagrant copy of TicTacToe, from game.py
    It's simplified, so that moves and utility are calculated as needed
    Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""


    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        # self.initial = GameState(to_move='Y', board={(1,1): 'Y', (2,1): 'Y', (3,1): 'Y',
        #                                              (4,2): 'B',(4,3): 'B',(4,4): 'B'})
        self.initial = GameState(to_move='Y', board={(1,1): 'Y', (2,1): 'Y', (3,1): 'Y',
                                                     (4,2): 'B', (4,3): 'B', (4,4): 'B'})


    def actions(self, state):
        moves = []
        player = state.to_move
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken adjacent to a valid piece as long as another piece isn't already there." \
        ""

        if player == 'B':
            for x in range(1, self.h + 2):
                for y in range(2, self.v + 2):
                    if state.board.get((x,y)) == 'B':
                        if(state.board.get((x-1,y)) == '.' or (x-1,y) not in state.board.keys()):
                            if (x - 1, y) not in moves:
                                moves.append((x-1,y))
                        if (x!=4):
                            if(y!= 4) and (state.board.get((x,y+1)) == '.' or (x,y+1) not in state.board.keys()):
                                if (x,y+1) not in moves:
                                    moves.append((x,y+1))
                            if (x != 4) and (state.board.get((x, y-1)) == '.' or (x, y-1) not in state.board.keys()):
                                if (y-1 != 1):
                                    if (x, y-1) not in moves:
                                        moves.append((x,y-1))

        if player == 'Y':
            for x in range(1, self.h+1):
                for y in range(1, self.v + 2):
                    if state.board.get((x,y)) == 'Y':
                        if (x,y+1) not in state.board.keys() or (state.board.get((x,y+1)) == '.'):
                            if (x,y+1) not in moves:
                                moves.append((x,y+1))
                        if (y!=1):
                            if(x-1 != 0):
                                if (x-1,y) not in state.board.keys() or (state.board.get((x-1,y)) == '.'):
                                    if (x-1,y) not in moves:
                                        moves.append((x-1,y))
                            if (x + 1, y) not in state.board.keys() or (state.board.get((x + 1, y)) == '.'):
                                if (x + 1 != 4):
                                    if (x + 1, y) not in moves:
                                        moves.append((x + 1, y))


        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'B':
            return 'Y'
        if player == 'Y':
            return 'B'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = state.board.copy()
        player = state.to_move
        board[move] = player
        if player == 'B':
            for x in range(0, self.h+2):
                for y in range(0, self.v+2):
                    if (x,y) == move:
                        if x == 0:
                            board[(x+1,y)] = '.'
                            board[(x,y)] = '.'
                        if ((x+1,y) in state.board.keys()) and (state.board.get((x+1, y))) == 'B':
                            board[(x+1,y)] = '.'
                        elif ((x,y-1) in state.board.keys()) and (state.board.get((x, y-1))) == 'B':
                            board[(x,y-1)] = '.'
                        elif ((x,y+1) in state.board.keys()) and (state.board.get((x, y+1))) == 'B':
                            board[(x,y+1)] = '.'


        elif player == 'Y':
            for x in range(0, self.h+2):
                for y in range(0, self.v+3):
                    if (x,y) == move:
                        if y == 5:
                            board[(x, y)] = '.'
                        if ((x,y-1) in state.board.keys()) and (state.board.get((x, y-1))) == 'Y':
                            board[(x,y-1)] = '.'
                        elif ((x-1,y) in state.board.keys()) and (state.board.get((x-1, y))) == 'Y':
                            board[(x-1,y)] = '.'
                        elif ((x+1,y) in state.board.keys()) and (state.board.get((x+1, y))) == 'Y':
                            board[(x+1,y)] = '.'
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
        n = 0
        for x in range(0, self.v + 2):
            for y in range(0, self.h + 2):
                if board.get((x,y)) == player:
                    n += 1
        return n == 0

    # does player have all their pieces off the board? Return true if more than one piece on board.
    # def board_piece_check(self, board, start, player):
    #     n=0
    #     for x in range (1, self.v + 1):
    #         for y in range (1, self.h + 1):
    #             if board == player:
    #                 n += 1
    #     return n == 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        b = 0
        for x in range (1, self.v + 1):
            for y in range (1, self.h + 1):
                if state.board.get((x, y)) == 'B':
                    b+=1
        y = 0
        for x in range(1, self.v + 1):
            for y in range(1, self.h + 1):
                if state.board.get((x, y)) == 'Y':
                    y += 1
        return ((b == 0) or (y == 0)) and len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 2):
                if y != 1:
                    print(board.get((x, y), '.'), end=' ')
                elif y == 1:
                    print(board.get((x, y), '.'), '|', end=' ')
            print()
        print('----------')
        x = 4
        for y in range(1, self.v + 2):
            if y != 1:
                print(board.get((x, y), '.'), end=' ')
            elif y == 1:
                print(board.get((x, y), '.'), '|', end=' ')
        print()
        print()




myGame = FlagrantCopy()

won = GameState(
    to_move = 'B',
    board = {
             (2,2): 'B', (2,3): 'B'
            },

    label = 'won'
)

winin1 = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'B',
             (2,1): 'Y', (2,2): 'Y',
            },
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'Y',
    board = {(1,1): 'B', (1,2): 'B',
             (2,1): 'Y', (2,2): 'Y',
             (3,1): 'B',
            },
    label = 'losein1'
)

winin3 = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'Y',
             (2,1): 'B',
             (3,1): 'Y',
            },
    label = 'winin3'
)

losein3 = GameState(
    to_move = 'Y',
    board = {(1,1): 'B',
             (2,1): 'B',
             (3,1): 'Y', (1,2): 'B', (1,2): 'Y',
            },
    label = 'losein3'
)

winin5 = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'Y',
             (2,1): 'B',
            },
    label = 'winin5'
)

lost = GameState(
    to_move = 'B',
    board = {(1,1): 'B', (1,2): 'B',
             (2,1): 'Y', (2,2): 'Y', (2,3): 'Y',
             (3,1): 'B'
            },
    label = 'lost'
)

myGames = {
    myGame: [
        #won,
        # winin1, losein1, winin3, losein3, winin5,
        # lost,
    ]
}


#
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# print('\n')
# winin1 = myGame.result(winin1,(1,3))
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# print('\n')
# winin1 = myGame.result(winin1,(1,2))
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# print('\n')
# winin1 = myGame.result(winin1,(1,4))
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# print('\n')
# winin1 = myGame.result(winin1,(0,2))
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# print('\n')
# winin1 = myGame.result(winin1,(1,2))
# myGame.display(winin1)
# print(myGame.terminal_test(winin1))
# print(myGame.actions(winin1))
# try_to_play(myGame)
