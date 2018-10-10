from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy
import random


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
        self.initial = GameState(to_move='X', board={})

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
        for y in range(1, self.v + 1):
            if self.k_in_row(board, (1,y), player, (1,0)):
                return 1
        # check columns
        for x in range(1, self.h + 1):
            if self.k_in_row(board, (x,1), player, (0,1)):
                return 1
        # check \ diagonal
        if self.k_in_row(board, (1,1), player, (1,1)):
            return 1
        # check / diagonal
        if self.k_in_row(board, (3,1), player, (-1,1)):
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


myGame = FlagrantCopy()

won = GameState(
    to_move = 'O',
    board = {(1,1): 'X', (1,2): 'X', (1,3): 'X',
             (2,1): 'O', (2,2): 'O',
            },
    label = 'won'
)

winin1 = GameState(
    to_move = 'X',
    board = {(1,1): 'X', (1,2): 'X',
             (2,1): 'O', (2,2): 'O',
            },
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
    to_move = 'X',
    board = {(1,1): 'X', (1,2): 'X',
             (2,1): 'O', (2,2): 'O', (2,3): 'O',
             (3,1): 'X'
            },
    label = 'lost'
)

class DiceState:    # one way to define the state of a minimal game.

    def __init__(self, player): # add parameters as needed.
        self.to_move = player
        self.label = str(id(self))   # change this to something easier to read
        # add code and self.variables as needed.

    def __str__(self):  # use this exact signature
        return self.label

# class TemplateAction:
#     '''
#     It is not necessary to define an action.
#     Start with actions as simple as a label (e.g., 'Down')
#     or a pair of coordinates (e.g., (1,2)).
#
#     Don't un-comment this until you already have a working game,
#     and want to play smarter.
#     '''
#     def __lt__(self, other):    # use this exact signature
#         # return True when self is a better move than other.
#         return False
# Rules: Pitcher will roll the dice. The algorithm easily explains the rules.
# Instead of 4 bases there are 3, First, Second, and Home. Players will pick roll
# at start. The Pitcher wins if he gets 5 outs and the Batter wins if he scores 5 points.
# (may have to tweak these numbers so that it's fair.)

class DiceBall(Game):

    def dice():
        return (random.randint(1,6))
    def batter():
        return (random.randint(1,6))

    print('Let\'s play a Game of Diceball!')
    print('The rules are as follows:\n')
    print('There is a pitcher and a hitter. The pitcher will roll and it will be decided whether\n'
          'it is a strike, ball, or if the batter gets to swing. If the batter gets to swing then\n'
          'the batter will roll and the roll will determine whether they strike or if they\n'
          'hit the ball. The first to 5 points wins!\n')

    player1 = 'Pitcher'
    player2 = 'Batter'
    player1Score = 0
    player2Score = 0
    strike = 0
    ball = 0
    foul = 0

    while player1Score != 5 and player2Score != 5:
        roll = dice()

        print('Pitcher throws ' + str(roll))
        if roll == 1 or roll == 2:
            print('Pitcher throws a strike!')
            strike += 1
            print("Strikes: " + str(strike))
            print("Balls: " + str(ball))
            print("Pitcher Score: " + str(player1Score))
            print("Batter Score: " + str(player2Score))
            if strike == 3:
                print('\nPitcher struck a batter out!')
                player1Score += 1
                print("Strikes: " + str(strike))
                print("Balls: " + str(ball))
                print("Pitcher Score: " + str(player1Score))
                print("Batter Score: " + str(player2Score))
        elif roll == 4 or roll == 3:
            print('Pitcher throws a ball!')
            ball += 1
            print("Strikes: " + str(strike))
            print("Balls: " + str(ball))
            print("Pitcher Score: " + str(player1Score))
            print("Batter Score: " + str(player2Score))
            if ball == 4:
                print('\nPitcher walks the Batter')
                player2Score += 1
                print("Strikes: " + str(strike))
                print("Balls: " + str(ball))
                print("Pitcher Score: " + str(player1Score))
                print("Batter Score: " + str(player2Score))
        elif roll == 5 or roll == 6:
            print('The Batter gets to swing!')
            swing = batter()
            if swing == 1 or swing == 2:
                print('The batter hits the ball and scores!')
                player2Score += 1
                print("Pitcher Score: " + str(player1Score))
                print("Batter Score: " + str(player2Score))
                strike = 0
                ball = 0
            elif swing == 3 or swing == 4:
                print('The batter hit a foul ball!')
                if strike != 3:
                    strike += 1
                    print("Strikes: " + str(strike))
                    print("Balls: " + str(ball))
                    print("Pitcher Score: " + str(player1Score))
                    print("Batter Score: " + str(player2Score))
                elif strike == 3:
                    strike -= 1
                    print("Strikes: " + str(strike))
                    print("Balls: " + str(ball))
                    print("Pitcher Score: " + str(player1Score))
                    print("Batter Score: " + str(player2Score))
            elif swing == 5 or swing == 6:
                print('The batter struck out!')
                player1Score +=1
                print("Strikes: " + str(strike))
                print("Balls: " + str(ball))
                print("Pitcher Score: " + str(player1Score))
                print("Batter Score: " + str(player2Score))
        print('\n')
        if player1Score == 5:
            print('The Pitcher Wins!!!!')
        elif player2Score == 5:
            print('The Batter wins!!!!')

#Commit



    '''
    This is a minimal Game definition,
    the shortest implementation I could run without errors.
    '''

    def __init__(self, initial):    # add parameters if needed.
        self.initial = initial

        # add code and self.variables if needed.

    def actions(self, state):   # use this exact signature.
        acts = []

        # append all moves, which are legal in this state,
        # to the list of acts.
        return acts

    def result(self, state, move):   # use this exact signature.
        newState = deepcopy(state)
        # use the move to modify the newState
        return newState

    def terminal_test(self, state):   # use this exact signature.
        # return True only when the state of the game is over.
        return True

    def utility(self, state, player):   # use this exact signature.
        ''' return:
        >0 if the player is winning,
        <0 if the player is losing,
         0 if the state is a tie.
        '''
        return 0

    def display(self, state):   # use this exact signature.
        # pretty-print the game state, using ASCII art,
        # to help a human player understand his options.
        print(state)

tg = DiceBall(DiceState('A'))   # this is the game we play interactively.

myGames = {
    DiceBall:[

    ]
    # myGame: [
    #     won,
    #     winin1, losein1, winin3, losein3, winin5,
    #     lost,
    # ],
    #
    # tg: [
    #     # these are the states we tabulate when we test AB(1), AB(2), etc.
    #     TemplateState('B'),
    #     TemplateState('C'),
    # ]
}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~Algorithm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules: Pitcher will roll the dice. The algorithm easily explains the rules.
# Instead of 4 bases there are 3, First, Second, and Home. Players will pick roll
# at start. The Pitcher wins if he gets 5 outs and the Batter wins if he scores 5 points.
# (may have to tweak these numbers so that it's fair.)
#
# pitcher rolls
# if(die=1 || die=2)
#     strike();
#     elif(die=4 || die=3)
#     ball();
#     else
#     batterswing();
# batter rolls
# if(die=1||die=2)
#     strike();
#     elif(die=4)
#     firstBase();
#     elif(die=5)
#     secondBase();
#     else
#     homeRun();