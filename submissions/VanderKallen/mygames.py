from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table

class GameState:
    def __init__(self, to_move, board, score, label=None):
        self.to_move = to_move
        self.board = board
        self.score = score
        self.label = label

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

    def actions(self, state):
        moves = []
        if state.score >= 29:
            self.terminal_test(state)
        else:
            if state.board == 1:
                moves.append(3)
                moves.append(4)
            else:
                if state.board == 2:
                    moves.append(5)
                    moves.append(4)
                else:
                    if state.board == 3:
                        moves.append(1)
                        moves.append(5)
                    else:
                        if state.board == 4:
                            moves.append(1)
                            moves.append(2)
                        else:
                            if state.board == 5:
                                moves.append(2)
                                moves.append(3)
                            else:
                                if state.board == 0:
                                    moves.append(1)
                                    moves.append(2)
                                    moves.append(3)
                                    moves.append(4)
                                    moves.append(5)
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'p1':
            return 'p2'
        if player == 'p2':
            return 'p1'
        return None

    def result(self, state, move):
        currMover = state.to_move
        nextMover = self.opponent(currMover)

        newState = deepcopy(state)
        newState.to_move = nextMover
        newState.board = move
        newState.score += move
        return newState

    def check_win(self, state):
        if state.score >= 29:
            return 1
        else:
            return 0

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        "if player goes over 29 they loose"
        if player == 'p1' and state.score >= 29:
            return 1
        if state.score < 29:
            return 0
        else:
            return -1

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return state.score >= 29

    def display(self, state):
        if state.board != 0:
            print()
            print('Score: ' + str(state.score))
            print('The coin is on the ' + str(state.board) + ' Spot.')
            print(str(state.to_move) +' may slide it to the ' + str(state.moves[0]) + ' space or the ' + str(state.moves[1]) + ' space.')
            print()
        else:
            print()
            print('Place your coin on any number to Start')

won = GameState(
    to_move = 'p1',
    board = 1,
    score = 29,
    label = 'won'
)

oneAway = GameState(
    to_move = 'p1',
    board = 1,
    score = 28,
    label = 'one away'
)

start = GameState(
    to_move = 'p1',
    board = 0,
    score = 0,
    label = 'start'
)

choice = GameState(
    to_move = 'p1',
    board = 1,
    score = 24,
    label = '4 is better'
)

choice2 = GameState(
    to_move = 'p1',
    board = 4,
    score = 27,
    label = '1 is better'
)

choice3 = GameState(
    to_move = 'p1',
    board = 3,
    score = 23,
    label = '5 is better'
)

choice4 = GameState(
    to_move = 'p1',
    board = 4,
    score = 27,
    label = 'choice4'
)
star29 = Star29(start)

myGames = {
    star29: [
        won,
        oneAway,
        start,
        choice,
        choice2,
        choice3,
        choice4
        #choice5
    ]
}