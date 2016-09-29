from collections import namedtuple
from games import (Game)
import copy

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

class Oink(Game):
    """A flagrant copy of TicTacToe, from game.py
    It's simplified, so that moves and utility are calculated as needed
    Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self):
        self.collCount = 4
        self.rowCount = 4
        self.piggyR = 0
        self.piggyC = 1
        self.fencesR = {'F1':3,
                       'F2':3,
                       'F3':3,
                       'F4':3}
        self.fencesC = {'F1':0,
                       'F2':1,
                       'F3':2,
                       'F4':3}
        self.initial = GameState(to_move='P', board=[ [' ',  'P',  ' ',  ' '],
                                                         [' ',  ' ',  ' ',  ' '],
                                                      [' ',  ' ',  ' ',  ' '],
                                                         ['F1', 'F2', 'F3', 'F4']])

    def actions(self, state):
        # try:
        #     return state.moves
        # except:
        #     pass
        # "Legal moves are any square not yet taken."
        moves = []
        if state.to_move == 'P':
            for i in range(self.rowCount):
                if i % 2 == 0 and 'P' in state.board[i]:
                    loc = state.board[i].index('P')
                    if i-1 >= 0:
                        if loc-1 >= 0 and state.board[i-1][loc-1] == ' ':
                            moves.append('P to (' + str(i-1) + ',' + str(loc-1) + ')')
                        if state.board[i-1][loc] == ' ':
                            moves.append('P to (' + str(i-1) + ',' + str(loc) + ')')
                    if i+1 < self.rowCount:
                        if loc-1 >= 0 and state.board[i+1][loc-1] == ' ':
                            moves.append('P to (' + str(i+1) + ',' + str(loc-1) + ')')
                        if state.board[i+1][loc] == ' ':
                            moves.append('P to (' + str(i+1) + ',' + str(loc) + ')')
                if i % 2 != 0 and 'P' in state.board[i]:
                    loc = state.board[i].index('P')
                    if i-1 >= 0:
                        if state.board[i-1][loc] == ' ':
                            moves.append('P to (' + str(i-1) + ',' + str(loc) + ')')
                        if loc+1 < self.collCount and state.board[i-1][loc+1] == ' ':
                            moves.append('P to (' + str(i-1) + ',' + str(loc+1) + ')')
                    if i+1 < self.rowCount:
                        if state.board[i+1][loc] == ' ':
                            moves.append('P to (' + str(i+1) + ',' + str(loc) + ')')
                        if loc+1 < self.collCount and state.board[i+1][loc+1] == ' ':
                            moves.append('P to (' + str(i+1) + ',' + str(loc+1) + ')')
        if state.to_move =='F':
            for k in self.fencesC.keys():
                for i in range(self.rowCount):
                    if i % 2 == 0 and k in state.board[i] and state.board[i][self.fencesC[k]] == k:
                        if i-1 >= 0:
                            if self.fencesC[k]-1 >= 0 and state.board[i-1][self.fencesC[k]-1] == ' ':
                                moves.append(k + ' to(' + str(i-1) + ',' + str(self.fencesC[k]-1) + ')')
                            if state.board[i-1][self.fencesC[k]] == ' ':
                                moves.append(k + ' to(' + str(i-1) + ',' + str(self.fencesC[k]) + ')')
                    if i % 2 != 0 and k in state.board[i] and state.board[i][self.fencesC[k]] == k:
                        if i-1 >= 0:
                            if state.board[i-1][self.fencesC[k]] == ' ':
                                moves.append(k + ' to(' + str(i-1) + ',' + str(self.fencesC[k]) + ')')
                            if self.fencesC[k]+1 < self.collCount and state.board[i-1][self.fencesC[k]+1] == ' ':
                                moves.append(k + ' to(' + str(i-1) + ',' + str(self.fencesC[k]+1) + ')')

        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'P':
            return 'F'
        if player == 'F':
            return 'P'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = copy.deepcopy(state.board)
        if state.to_move == 'P':
            player = 'P'
            retPlayer = 'P'
        else:
            player = move[:2]
            retPlayer = 'F'
        tempR = int(move[6:-3])
        tempC = int(move[8:-1])
        board[tempR][tempC] = player
        if state.to_move == 'P':
            board[self.piggyR][self.piggyC] = ' '
            self.piggyR = tempR
            self.piggyC = tempC
        else:
            board[self.fencesR[player]][self.fencesC[player]] = ' '
            self.fencesC[player] = tempC
            self.fencesR[player] = tempR
        next_mover = self.opponent(retPlayer)
        return GameState(to_move=next_mover, board=board)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        try:
            return state.utility if player == 'P' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, state, 'P')
        if util == 0:
            util = -self.check_win(board, state, 'F')
        state.utility = util
        return util if player == 'P' else -util

    # Did I win?
    def check_win(self, board, state, player):
        met = 0
        if player == 'P' and self.piggyR == 3:
            return 1
        if player == 'F':
            chkActions = copy.deepcopy(state)
            chkActions.to_move ='P'
            chkMoves = self.actions(chkActions)
            if len(chkMoves) == 0:
                return 1
        return 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'P') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(self.rowCount):
            for y in range(self.collCount):
                if x % 2 == 0:
                    if board[x][y] in self.fencesC.keys():
                        pStr = board[x][y] + '##'
                    else:
                        pStr = board[x][y] + ' ##'
                if x % 2 != 0:
                    if board[x][y] in self.fencesC.keys():
                        pStr = '##' + board[x][y]
                    else:
                        pStr = '## ' + board[x][y]
                print(pStr, end='')
            print()


myGame = Oink()

won = GameState(
    to_move = 'F',
    board = [ [' ',   ' ',  'F1',   ' '],
                 ['F2',  ' ',   'F4',  ' '],
              [' ',   ' ',   ' ',   ' '],
                 ['P',   'F3',   ' ',  ' ']],
    label = 'won'
)

winin1 = GameState(
    to_move = 'X',
    board = [ [' ',' ','F1',' '],
                  ['F2','P','F4',' '],
              [' ',' ','F3',' '],
                  [' ',' ',' ',' ']],
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'O',
    board = {(1,1): 'X', (1,2): 'X',
             (2,1): 'O', (2,2): 'O',
             (3,1): 'X',
            },
    label = 'losein1'
)

winin3 = GameState(
    to_move = 'X',
    board = {(1,1): 'X', (1,2): 'O',
             (2,1): 'X',
             (3,1): 'O',
            },
    label = 'winin3'
)

losein3 = GameState(
    to_move = 'O',
    board = {(1,1): 'X',
             (2,1): 'X',
             (3,1): 'O', (1,2): 'X', (1,2): 'O',
            },
    label = 'losein3'
)

winin5 = GameState(
    to_move = 'X',
    board = {(1,1): 'X', (1,2): 'O',
             (2,1): 'X',
            },
    label = 'winin5'
)

lost = GameState(
    to_move = 'P',
    board = [ [' ',  'F1',  'F4',  ' '],
                 [' ',  'P',  ' ',  ' '],
              [' ',  'F2',  'F3',  ' '],
                 [' ', ' ', ' ', ' ']],
    label = 'lost'
)

myGames = {
    myGame: [
        won,
        winin1, losein1, winin3, losein3, winin5,
        lost,
    ]
}