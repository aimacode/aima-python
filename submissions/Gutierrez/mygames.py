from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table

class GameState:    # one way to define the state of a minimal game.

    def __init__(self, to_move, bag1, bag2, bag3, label): # add parameters as needed.
        self.to_move = to_move
        self.label = label
        self.bag1 = bag1
        self.bag2= bag2
        self.bag3= bag3
        self.scores = {'Alpha': 0, 'Beta': 0}
        # change this to something easier to read
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

    def __init__(self, state):    # add parameters if needed.
        self.initial = state
        # add code and self.variables if needed.

    def actions(self, state):   # use this exact signature.
        acts = []
        c=0
        for i in state.bag1:
            c +=1
            acts.append("bag1 "+str(c))
        c=0
        for i in state.bag2:
            c +=1
            acts.append("bag2 "+str(c))
        c=0
        for i in state.bag3:
            c +=1
            acts.append("bag3 "+ str(c))

        # append all moves, which are legal in this state,
        # to the list of acts.
        return acts

    def opponent(self, player):
        if player == 'Alpha':
            return 'Beta'
        if player == 'Beta':
            return 'Alpha'
        return None

    def result(self, state, move):   # use this exact signature.
        newState = deepcopy(state)
        currMover = state.to_move
        nextMover = self.opponent(currMover)

        splice = move.split()
        num= splice[-1]
        c = int(num)
        bag = splice[1]
        if bag == ("bag1"):
            i= 0
            while i < c:
                del(state.bag1[i])
                newState.bag1 = state.bag1
                i+=1
        if bag == "bag2":
            i = 0
            while i < c:
                del (state.bag2[i])
                newState.bag2 = state.bag2
                i += 1
        if bag== "bag3":
            i = 0
            while i < c:
                del (state.bag3[i])
                newState.bag3 = state.bag3
                i += 1
        global stateCount
        stateCount += 1

        newState.to_move = nextMover
#        newState.scores[currMover] += v
        return newState
        # use the move to modify the newState

        return newState

    def terminal_test(self, state):   # use this exact signature.
        if len(state.bag1)==0 and len(state.bag2)==0 and len(state.bag3)== 0:
            currMover = state.to_move
            state.score[currMover]+= 1
            return True
        else: return False

    def utility(self, state, player):   # use this exact signature.
        ''' return:
        >0 if the player is winning,
        <0 if the player is losing,
         0 if the state is a tie.
        '''
        opponent = self.opponent(player)
        return state.scores[player] - state.scores[opponent]

    def display(self, state):
        print (state.bag1, state.bag2, state.bag3)
        print("statecount")
        print(stateCount)

        # use this exact signature.
        # pretty-print the game state, using ASCII art,
        # to help a human player understand his options.
        print(state)
stateCount = 1
stolen = GameState(
    to_move = 'Alpha',

    bag1 =[1, 1 ,1,1,1],
    bag2 = [1,1 ,1],
    bag3 =[1, 1 ,1,1],
    label = 'stolen'
)
won = GameState(
    to_move = 'Alpha',

    bag1 =[],
    bag2 = [1,1 ,1],
    bag3 =[],
    label = 'won'
)
lost = GameState(
    to_move = 'Beta',

    bag1 =[],
    bag2 = [1,1 ,1],
    bag3 =[],
    label = 'lost'
)
oneToWin = GameState(
    to_move = 'Beta',

    bag1 =[1],
    bag2 = [1],
    bag3 =[],
    label = 'oneToWin'
)
oneToLose = GameState(
    to_move = 'Alpha',

    bag1 =[1],
    bag2 = [1],
    bag3 =[],
    label = 'oneToWin'
)
# tg = TemplateGame(TemplateState('A'))   # this is the game we play interactively.
myGame = TemplateGame(stolen)
myGames = {
    myGame: [
        stolen,
        won,
        lost,
        oneToWin,
    ]
}