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

class CTT(Game):
    """ A really modified version of Circle-Tic-Tac-Toe
    In this game the goal is to get either a 3-in-a-row diagonally or across in the center
    or get a 5-in-a-row that forms a circle. There are 5 places to get a 3-in-a-row and there
    are 3 places to get a 5-in-a-row. The board looks like this
      1 2 3 4 5 6 7
    1|# . . . . . #
    2|. # . . . # .
    3|. . # . # . .
    4|. . . . . . .
    5|. . # # # . .
    6|. # . # . # .
    7|# . . # . . #
    So (1,1) (7,1) (4,7) (7,7) (1,7) (1,1) would make a 5-in-a-row circle
    and (1,1) (2,2) (3,3) or (5,3) (5,4) (5,5) would make a 3-in-a-roww
    """

    def __init__(self, h=7, v=7, k=3):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={})

        #Valid spaces make up the spaces within the 7x7 grid that are "allowable".
        #This is set as an immutable Tuple since it will be true always and can't be changed.
        self.validSpaces = (
        (1, 1), (1, 7), (2, 2), (2, 6), (3, 3), (3, 5), (5, 4), (6, 4),(7,4),
        (7,1),(6,2),(5,3),(5,5),(6,6),(7,7))
        #c1, c2, c3, d1, d2, d3, d4, d5, and r1 are the coordinates that make up the 9 winning states.
        #these will be checked when determining if the game is over.
        #c stands for 5-in-a-row circle wins
        #d stands for 3-in-a-row diagonal wins
        #r stands for 3-in-a-row row win for the center
        self.c1 = ((3,3), (3,5), (5,4),(5,3),(5,5))
        self.c2 = ((2,2), (2,6), (6,4),(6,2), (6,6))
        self.c3 = ((1,1), (1,7), (7,4), (7,1), (7,7))
        self.d1 = ((1,1), (2,2), (3,3), )
        self.d2 = ((1,7), (2,6), (3,5),)
        self.d3 = ((5,4), (6,4), (7,4),)
        self.d4 = ((7,1),(6,2),(5,3))
        self.d5 = ((7, 7), (6, 6), (5, 5))
        self.r1 = ((5,3),(5,4),(5,5))



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
        # self.display (state)
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
        if self.k_in_row(self.d4,board,player) == 3:
         return 1
        if self.k_in_row(self.d5,board,player) == 3:
         return 1
        if self.k_in_row(self.r1,board,player) == 3:
         return 1
        # check c1
        if self.k_in_row(self.c1,board,player) == 5:
         return 1
        # check c2
        if self.k_in_row(self.c2,board,player) == 5:
         return 1
        # check c3
        if self.k_in_row(self.c3,board,player) == 5:
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
    board = {(3,3): 'X', (3,5): 'X',(5,3): 'X',(5,4): 'X', (5,5): 'X',
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


myGames = {
    myGame: [
        won,
        winin1, losein1,
        winin2, stalemate, new
    ]
}

