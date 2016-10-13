from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table

class GameState:
    def __init__(self, to_move, position, score, label=None):
        self.to_move = to_move
        self.position = position
        self.label = label
        self.score = score

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Star29(Game):
    """
    An implementation of ThinkAhead
    """
    def __init__(self, state):
        self.initial = state

    def availableMoves(self, state):
        moves = []
        if state.score >= 29:
            moves = []
        else:
            if state.position == 1:
                moves.append(3)
                moves.append(4)
            else:
                if state.posititon == 2:
                    moves.append(4)
                    moves.append(5)
                else:
                    if state.position == 3:
                        moves.append(1)
                        moves.append(5)
                    else:
                        if state.position == 4:
                            moves.append(1)
                            moves.append(2)
                        else:
                            if state.position == 5:
                                moves.append(2)
                                moves.append(3)
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'Player1':
            return 'Player2'
        if player == 'Player1':
            return 'Player2'
        return None

    def result(self, state, move):
        currMover = state.to_move
        nextMover = self.opponent(currMover)
        newState = deepcopy(state)
        newState.to_move = nextMover
        newState.position = move
        newState.score += move
        return newState

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        if state.score >= 29:
            return 1
        else:
            return 0

    def utility(self, state, player):
        "Check and go"
        test = state.terminal_test(state)
        if test == 1:
            return -1
        else:
            if test == 0:
                return 0

    def display(self, state):
        print(str(state.board))
        print('Score: ' + str(state.score))


start = GameState(
    to_move = 'Player1',
    position = 1,
    score = 0,
    label = 'start'
)

star29 = Star29(start)

myGames = {
    star29: [
        start
    ]
}