import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))

# HW 5 Custom map
wayne_map = search.UndirectedGraph(dict(
    Plymouth=dict(Livonia=11, Westland=17),
    Livonia=dict(Plymouth=14, Hamtramck=26, Detroit=23, Garden_City=10),
    Hamtramck=dict(Livonia=26, Detroit=12),
    Westland=dict(Plymouth=17, Garden_City=6, Wayne=10),
    Garden_City=dict(Livonia=10, Westland=6, Dearborn=16),
    Detroit=dict(Hamtramck=12, Livonia=23, Dearborn=13, Lincoln_Park=21),
    Wayne=dict(Westland=10, Dearborn=20, Romulus=12),
    Dearborn=dict(Garden_City=16, Detroit=13, Lincoln_Park=21, Wayne=20, Taylor=15),
    Lincoln_Park=dict(Detroit=21, Dearborn=11, Wyandotte=8),
    Belleville=dict(Romulus=11),
    Romulus=dict(Belleville=11, Wayne=12, Taylor=13),
    Taylor=dict(Romulus=13, Dearborn=15, Woodhaven=11),
    Wyandotte=dict(Lincoln_Park=8, Trenton=15),
    Flat_Rock=dict(Woodhaven=8),
    Woodhaven=dict(Flat_Rock=8, Taylor=11, Trenton=13),
    Trenton=dict(Woodhaven=13, Wyandotte=15)
))

# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
myPuzzle = search.GraphProblem('Livonia', 'Belleville', wayne_map)

myPuzzle.label = 'Wayne County'
myPuzzle.description = '''
An abbreviated map of Wayne County, MI.
This map is unique, to the best of my knowledge.
'''

from grid import distance

wayne2map = dict(
    Plymouth=dict(Livonia=11, Westland=17),
    Livonia=dict(Plymouth=14, Hamtramck=26, Detroit=23, Garden_City=10),
    Hamtramck=dict(Livonia=26, Detroit=12),
    Westland=dict(Plymouth=17, Garden_City=6, Wayne=10),
    Garden_City=dict(Livonia=10, Westland=6, Dearborn=16),
    Detroit=dict(Hamtramck=12, Livonia=23, Dearborn=13, Lincoln_Park=21),
    Wayne=dict(Westland=10, Dearborn=20, Romulus=12),
    Dearborn=dict(Garden_City=16, Detroit=13, Lincoln_Park=21, Wayne=20, Taylor=15),
    Lincoln_Park=dict(Detroit=21, Dearborn=11, Wyandotte=8),
    Belleville=dict(Romulus=11),
    Romulus=dict(Belleville=11, Wayne=12, Taylor=13),
    Taylor=dict(Romulus=13, Dearborn=15, Woodhaven=11),
    Wyandotte=dict(Lincoln_Park=8, Trenton=15),
    Flat_Rock=dict(Woodhaven=8),
    Woodhaven=dict(Flat_Rock=8, Taylor=11, Trenton=13),
    Trenton=dict(Woodhaven=13, Wyandotte=15)
)

wayne2Locations = dict(
    Plymouth=(42.371426, 83.470215),
    Livonia=(42.368370, 83.352710),
    Hamtramck=(42.392815, 83.049644),
    Westland=(42.323806, 83.400532),
    Garden_City=(42.325592, 83.331039),
    Detroit=(42.331429, 83.045753),
    Wayne=(42.268241, 83.284417),
    Dearborn=(42.322262, 83.176315),
    Lincoln_Park=(42.250594, 83.178536),
    Belleville=(42.204841, 83.485211),
    Romulus=(42.222261, 83.396600),
    Taylor=(42.240872, 83.269651),
    Wyandotte=(42.200662, 83.151016),
    Flat_Rock=(42.096431, 83.291874),
    Woodhaven=(42.137052, 83.245107),
    Trenton=(42.140655, 83.180054)
)


class Wayne2(search.Problem):
    # map = rmap

    def __init__(self, map, locations, start, finish):
        self.map = map
        self.locations = locations
        self.initial = start
        self.finish = finish

    def actions(self, state):
        neighbors = self.map[state]
        keys = neighbors.keys()
        return keys
        # return list(keys)

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.finish

    def path_cost(self, c, state1, action, state2):
        neighbors = self.map[state1]
        cost = neighbors[state2]
        return c + cost

    def h(self, node):
        state = node.state
        loc1 = self.locations[state]
        loc2 = self.locations[self.finish]
        return distance(loc1, loc2)


wayne_puzzle2 = Wayne2(wayne2map, wayne2Locations, 'Garden_City', 'Wayne')
wayne_puzzle2.label = 'Wayne2'
wayne_puzzle3 = Wayne2(wayne2map, wayne2Locations, 'Detroit', 'Belleville')
wayne_puzzle3.label = 'Wayne3'


romania_map = search.UndirectedGraph(dict(
    A=dict(Z=75,S=140,T=118),
    Z=dict(O=71,A=75),
    S=dict(O=151,R=80,F=99),
    T=dict(A=118,L=111),
    O=dict(Z=71,S=151),
    L=dict(T=111,M=70),
    M=dict(L=70,D=75),
    D=dict(M=75,C=120),
    R=dict(S=80,C=146,P=97),
    C=dict(R=146,P=138,D=120),
    F=dict(S=99,B=211),
    P=dict(R=97,C=138,B=101),
    B=dict(G=90,P=101,F=211),
))

romania_puzzle = search.GraphProblem('A', 'B', romania_map)

romania_puzzle.label = 'Romania'
romania_puzzle.description = '''
The simplified map of Romania, per
Russall & Norvig, 3rd Ed., p. 68.
'''


# A trivial Problem definition
class LightSwitch(search.Problem):
    def actions(self, state):
        return ['up', 'down']

    def result(self, state, action):
        if action == 'up':
            return 'on'
        else:
            return 'off'

    def goal_test(self, state):
        return state == 'on'

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


# Twiddle game, a 3x3 board where inner squares of 4 tiles are rotated counter-clockwise to try and arrange the
# numbers in order.
# twiddle = '9,8,5,4,1,6,2,3,7'
# twiddle_goal = '1,2,3,4,5,6,7,8,9'
twiddle = ['9','8','5','4','1','6','2','3','7']
twiddle_goal = ['1','2','3','4','5','6','7','8','9']


def state2String(myState):
    strState = ','.join(myState)
    return strState


def string2State(myString):
    myState = myString.split(',')
    return myState

class  Twiddle(search.Problem):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        return ['upperLeft', 'lowerLeft', 'upperRight', 'lowerRight']

    def result(self, state, action):
        # myState = string2State(state)
        myState = state
        if action == 'upperLeft':
            temp = [myState[0], myState[1], myState[3], myState[4]]
            myState[3] = temp[0]
            myState[0] = temp[1]
            myState[4] = temp[2]
            myState[1] = temp[3]
        elif action == 'lowerLeft':
            temp = [myState[3], myState[4], myState[6], myState[7]]
            myState[6] = temp[0]
            myState[3] = temp[1]
            myState[7] = temp[2]
            myState[4] = temp[3]
        elif action == 'upperRight':
            temp = [state[1], state[2], state[4], state[5]]
            myState[4] = temp[0]
            myState[1] = temp[1]
            myState[5] = temp[2]
            myState[2] = temp[3]
        elif action == 'lowerRight':
            temp = [state[4], state[5], state[7], state[8]]
            myState[7] = temp[0]
            myState[4] = temp[1]
            myState[8] = temp[2]
            myState[5] = temp[3]
        # strState = state2String(myState)
        return myState

    def goal_test(self, state):
        return state == self.goal

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


# swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
# switch_puzzle = LightSwitch('off')
# switch_puzzle.label = 'Light Switch'
twiddle_puzzle = Twiddle(twiddle, twiddle_goal)
twiddle_puzzle.label = 'Twiddle'

mySearches = [
 #   swiss_puzzle,
    myPuzzle,
    romania_puzzle,
    # switch_puzzle,
    # twiddle_puzzle,
    wayne_puzzle2,
    wayne_puzzle3
]

mySearchMethods = []
