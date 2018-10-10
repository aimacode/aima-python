from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy


class GameState:
    def __init__(self, to_move, board, label=None, depth=8, score=0):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = depth
        self.score = score

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label


class Star29(Game):
    def __int__(self):
        self.initial = GameState(to_move=1, board=[1], score=0)
        #self.initial = GameState(to_move='1', board=['1'], score=0)

    def actions(self, state):
        board = state.board
        possible_moves = {1: [3, 4], 2: [4, 5], 3: [1, 5], 4: [1, 2], 5: [2, 3]}
        #possible_moves = {'1': ['3', '4'], '2': ['4', '5'], '3': ['1', '5'], '4': ['1', '2'], '5': ['2', '3']}
        current = board[-1]
        moves = possible_moves[current]
        state.moves = moves
        return moves

    def opponent(self, player):
        if player == 1:
            return 2
        if player == 2:
            return 1
        return None
    
    def result(self, state, move):
        board = state.board.copy()
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        score = state.score
        player = state.to_move
        points = int(move)
        score = score + points
        board.append(move)
        next_mover = self.opponent(player)
        gs = GameState(to_move=next_mover, board=board, score=score)
        return gs
    
    def utility(self, state, player):
        score = state.score
        util = score - 29
        state.utility = util
        return util

    def terminal_test(self, state):
        """If a player reaches 29 points, they loose."""
        '''
        player = state.to_move
        tt = self.utility(state, player) >= 0
        return tt
        '''
        if state.score >= 29:
            return 1
        return 0
    
    def display(self, state):
        board = state.board
        board_string = '['
        for i in range(len(board)):
            board_string = board_string + str(board[i]) + ', '
        board_string = board_string[:-2] + ']'
        print('Board: {1: [3, 4], 2: [4, 5], 3: [1, 5], 4: [1, 2], 5: [2, 3]}')
        print('Points: ' +  str(state.score) )
        print('Numbers so far: ' + board_string)
        print('Next player: ' + str(state.to_move))
        print()
        #print(state.board)


myGame = Star29()

won = GameState(
    to_move='1',
    board=[1, 3, 5, 2, 4, 1, 3, 5, 3, 5],
    #board=['1', '3', '5', '2', '4', '1', '3', '5', '3', '5'],
    label='won',
    score=31
)

winin1 = GameState(
    to_move='2',
    board=[1, 3, 5, 2, 4, 1, 3, 5, 3],
    #board=['1', '3', '5', '2', '4', '1', '3', '5', '3'],
    label='winin1',
    score=26
)

losein1 = GameState(
    to_move=1,
    board=[1, 4, 2, 5, 3, 1, 3, 5, 3, 1],
    #board=['1', '4', '2', '5', '3', '1', '3', '5', '3', '1'],
    label='losein1',
    score=27
)

winin3 = GameState(
    to_move=1,
    board=[1, 3, 5, 2, 4, 1, 3],
    #board=['1', '3', '5', '2', '4', '1', '3'],
    label='winin3',
    score=18
)

losein3 = GameState(
    to_move=2,
    board=[1, 4, 2, 5, 3, 1, 3, 5],
    label='losein3',
    score=23
)

lost = GameState(
    to_move=1,
    board=[5],
    label='lost',
    score=29
)

game = GameState(to_move=1, board=[1], score=0)

class TemplateState:    # one way to define the state of a minimal game.

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

class TemplateGame(Game):
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

tg = TemplateGame(TemplateState('A'))   # this is the game we play interactively.

myGames = {
    myGame: [
        won,
        winin1,
        losein1,
        winin3,
        losein3,
        #winin5,
        lost,
    ],


    tg: [
        # these are the states we tabulate when we test AB(1), AB(2), etc.
        TemplateState('B'),
        TemplateState('C'),
    ]
}
