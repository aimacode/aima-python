from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy

class GameState:
    def __init__(self, to_move, p1_hand, p2_hand, label=None):
        self.to_move = to_move
        self.p1_hand = p1_hand
        self.p2_hand = p2_hand
        self.label = label

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Chopsticks(Game):

    def __init__(self, initial):
        self.p1_hand = [1, 1]
        self.p2_hand = [1, 1]
        self.initial = initial

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        moves = []
        if (state.p1_hand[0] == 0 and state.p1_hand[1] == 0) or (state.p2_hand[0] == 0 and state.p2_[1] == 0):
            moves = []
        elif state.p1_hand[0] != 0:
            moves.append((state.p2_hand[0] + state.p1_hand[0]) % 5)
            moves.append((state.p2_hand[1] + state.p1_hand[0]) % 5)
        elif state.p1_hand[1] != 0:
            moves.append((state.p2_hand[0] + state.p1_hand[1]) % 5)
            moves.append((state.p2_hand[1] + state.p1_hand[1]) % 5)
        elif state.p2_hand[0] != 0:
            moves.append((state.p2_hand[0] + state.p2_hand[0]) % 5)
            moves.append((state.p2_hand[1] + state.p2_hand[0]) % 5)
        elif state.p2_hand[1] != 0:
            moves.append((state.p2_hand[0] + state.p2_hand[1]) % 5)
        elif state.p2_hand[0] != 0:
            moves.append((state.p1_hand[0] + state.p2_hand[0]) % 5)
            moves.append((state.p1_hand[1] + state.p2_hand[0]) % 5)
        elif state.p2_hand[1] != 0:
            moves.append((state.p1_hand[0] + state.p2_hand[1]) % 5)
            moves.append((state.p1_hand[1] + state.p2_hand[1]) % 5)
        elif state.p1_hand[0] != 0:
            moves.append((state.p1_hand[0] + state.p1_hand[0]) % 5)
            moves.append((state.p1_hand[1] + state.p1_hand[0]) % 5)
        elif state.p1_hand[1] != 0:
            moves.append((state.p1_hand[0] + state.p1_hand[1]) % 5)
        state.moves = moves
        moves = PriorityQueue()
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'p1':
            return 'p2'
        if player == 'p2':
            return 'p1'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state
        p1_hand = state.p1_hand.copy()
        p2_hand = state.p2_hand.copy()
        player = state.to_move
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, p1_hand = state.p1_hand, p2_hand = state.p2_hand)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        try:
            return state.utility if player == 'p1' else -state.utility
        except:
            pass
        util = self.check_win(state)
        if util == 0:
            util = -self.check_win(state)
        state.utility = util
        return util if player == 'p1' else -util

    def check_win(self, state):
        if state.p1_hand == [0, 0]:
            return 1
        else:
            return 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return state.p1_hand == [0, 0]

    def display(self, state):
        print('P1 left =', state.p1_hand[0], ' P1 right =', state.p1_hand[1],
              ' P2 left =', state.p2_hand[0], ' P2 right =', state.p2_hand[1])


myGame = Chopsticks()

won = GameState(
    to_move = 'P1',
    p1_hand = [0, 0],
    label = 'won'
)

lost = GameState(
    to_move = 'P2',
    p2_hand = [0, 0],
    label = 'lost'
)

myGames = {
    myGame: [
        won,
        lost,
    ]
}