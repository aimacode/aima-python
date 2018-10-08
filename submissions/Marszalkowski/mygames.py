from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy

class reverseNimState:    # one way to define the state of a minimal game.

    def __init__(self, to_move, board, label=None, depth=8):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = depth

    def __str__(self):
        if self.label == None:
            return super(reverseNimState, self).__str__()
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

class reverseNim(Game):
    '''
    This is a minimal Game definition,
    the shortest implementation I could run without errors.
    '''

    def __init__(self, piles=[3, 4, 5]):    # add parameters if needed.
        self.piles = piles
        self.initial = reverseNimState(to_move='A', board=[3, 4, 5])
        # add code and self.variables if needed.

    def actions(self, state):   # use this exact signature.
        acts = []
        for x in range(1, state.board[0] + 1):
            acts.append((1, x))
        for y in range(1, state.board[1] + 1):
            acts.append((2, y))
        for z in range(1, state.board[2] + 1):
            acts.append((3, z))

        state.acts = acts
        return acts

    def opponent(self, player):
        if player == 'A':
            return 'B'
        if player == 'A':
            return 'B'
        return None

    def result(self, state, move):   # use this exact signature.
        '''
        #newState = deepcopy(state)
        if move not in self.actions(state):
            return state
        player = state.to_move
        pile, amount = move
        self.piles[pile-1] -= amount
        board = state.board.copy()
        next_mover = self.opponent(player)
        return reverseNimState(to_move=next_mover, board=board)
        # use the move to modify the newState
        #return newState
        '''

        newState = deepcopy(state)
        pile, amount = move
        player = state.to_move
        next_mover = self.opponent(player)
        newState.to_move = next_mover
        if newState.board[pile-1]- amount >= 0:
            newState.board[pile-1] -= amount
        return newState


    def terminal_test(self, state): # use this exact signature.
        # return True only when the state of the game is over.
        if state.board == [0, 0, 0]:
            return True

    def utility(self, state, player):   # use this exact signature.
        '''
        try:
            return state.utility if player == 'A' else -state.utility
        except:
            pass
        '''
        board = state.board
        player = state.to_move
        if board == [1, 0, 0]:
            util = -1
        elif board == [0, 1, 0]:
            util = -1
        elif board == [0, 0, 1]:
            util = -1
        elif board[0] == 0 and board[1] == 0 and board[2] != 0:
            util = 1
        elif board[0] == 0 and board[1] != 0 and board[2] == 0:
            util = 1
        elif self.terminal_test(state) == True:
            util = -1
        else:
            util = 1
        return util if player == 'A' else -util




    def display(self, state):   # use this exact signature.
        board = state.board
        print(board)

rn = reverseNim()

loss = reverseNimState(
    to_move = 'B',
    board = [0, 0, 0],
    label = 'lost'
)

myGames = {

    rn: [
        loss
        # these are the states we tabulate when we test AB(1), AB(2), etc.
    ]
}