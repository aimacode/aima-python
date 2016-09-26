from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table

class GameState:
    def __init__(self, to_move, position, board, label=None):
        self.to_move = to_move
        self.position = position
        self.board = board
        self.label = label
        self.scores = {'S': 0}

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Move:
    def __init__(self, p, v):
        self.position = p
        self.value = v
        self.initial = GameState(to_move='Player One')
    def pv(self):
        return self.position, self.value
    # def getValue(self):

class Star29(Game):
    """
    An implementation of ThinkAhead
    """
    def __init__(self, state):
        self.initial = state
        self.first = ''
        self.startingBoard = state.board

    def actions(self, state):
        ""
        p = state.position

        return state.board

    # defines the order of play
    def opponent(self, player):
        if player == 'Player One':
            return 'Player Two'
        if player == 'Player One':
            return 'Player Two'
        return None

    def result(self, state, move):
        p = move
        # state.board[p] = v
        currMover = state.to_move
        nextMover = self.opponent(currMover)
        value = move
        # state.board[0] +
        newState = deepcopy(state)
        newState.to_move = nextMover
        newState.position = p
        index =  newState.board.index(value)
        if value == self.startingBoard[0]:
            newState.board = (self.startingBoard[2], self.startingBoard[3])
        if value == self.startingBoard[1]:
            # newState.board = (4,5)
            newState.board = (self.startingBoard[3], self.startingBoard[4])
        if value == self.startingBoard[2]:
            # newState.board = (5,1)
            newState.board = (self.startingBoard[4], self.startingBoard[0])
        if value == self.startingBoard[3]:
            # newState.board = (1, 2)
            newState.board = (self.startingBoard[0], self.startingBoard[1])
        if value == self.startingBoard[4]:
            # newState.board = (2, 3)
            newState.board = (self.startingBoard[1], self.startingBoard[2])
        # newState.board[p] = nan
        newState.scores['S'] += value


        self.lastMove = newState.position
        return newState

    def utility(self, state, player):
        "Player relative score"
        if state.scores['S'] > 29:
            return 1
        elif state.scores['S'] == 29:
            return -1
        else:
            return 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        if state.scores['S'] >= 29:
            return -1

        return 0

    def display(self, state):
        print(state.board)
        print('Score: ' + str(state.scores))

    # def check_win(self, board, player, state):
    #     if state.scores['P2'] >= 29 and player=="Player Two" :
    #         return 1
    #     if state.scores['P1'] >= 29 and player == "Player One":
    #         return -1
    #
    #     return 0


full_game = GameState(
    to_move = 'Player One',
    position = 0,
    board=[1,2,3,4,5],
    label = 'full'
)
full_game.scores = {'S':27}
mid_game = GameState(
    to_move = 'Player One',
    position = 0,
    board=[1,2,3,4,5],
    label = 'mid'
)
mid_game.scores = {'S':15}

won_game = GameState(
    to_move = 'Player One',
    position = 0,
    board=[1,2,3,4,5],
    label = 'won'
)
won_game.scores = {'S':27}

thinkA = Star29(full_game)

myGames = {
    thinkA: [
        full_game,
        mid_game,
        won_game
    ]
}