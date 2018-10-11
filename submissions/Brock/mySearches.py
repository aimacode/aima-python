import search
from math import(cos, pi)

# # A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))
#
# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
#
# sumner_puzzle.label = 'Sumner'
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''

ashgabat_map = search.UndirectedGraph(dict(
    Kommunizm=dict(Bezmein=10, Bagyr=14, Pilmile=60),
   Pewrize=dict(Bagyr=10, Shirvan=100, Faruj=130),
   Bagyr=dict(Bezmein=8, Kipchak=9, Pewrize=10, Kommunizm=14),
   Bezmein=dict(Bagyr=8, Kipchak=5, Kommunizm=10),
   Kipchak=dict(Bezmein=5, Bagyr=9),
    Shirvan=dict(Pewrize=100, Bojnourd=50, Faruj=42),
    Faruj=dict(Shirvan=42, Pewrize=130, Bojnourd=98),
    Bojnourd=dict(Faruj=98, Shirvan=50, Pilmile=50),
    Pilmile=dict(Bojnourd=50, Kommunizm=60),

))

ashgabat_puzzle = search.GraphProblem('Bojnourd', 'Kipchak', ashgabat_map)

ashgabat_puzzle.label = 'Ashgabat'
ashgabat_puzzle.description = '''
An abbreviated map of Ashgabat, Turkmenistan.
This map is unique, to the best of my knowledge.
'''

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

SinglesInitState = [[0,0,3], [0,1,4], [1,0,3], [1,1,5]]

class Singles(search.Problem):
    def __init__(self, initial):
        self.width = 2
        self.height = 2
        self.initial = initial
    def actions(self, state):
        return [[0,0, 0], [0,1,1],[1,0,2], [1,1,3]]

    # def state2String(self, myState):
    #     answer = ''
    #     for x in myState:
    #         for y in x:
    #             answer += y + ','
    #         answer = answer[:-1]
    #         answer += '|'
    #     return answer[:-1]
    # def string2State(self, myString):
    #     state = myString.split('|')
    #     count = 0
    #     for x in state:
    #         state[count] = x.split(',')
    #         count += count
    #     return state

    # def searchAction(self, x, y):
    #     return

    def result(self, state, action):
        if action[0]-1 != -1 and state[action[2]-1][2] != 0:
            return state
        if action[0]+1 != self.width and state[action[2]+1][2] != 0:
            return state
        if action[1]-1 != -1 and state[action[2]-self.width][2] != 0:
            return state
        if action[1]+1 != self.height and state[action[2]+self.width][2] != 0:
            return state
        state[action[2]][2] = 0
        return state

    def goal_test(self, state):
        return state == state

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

singles_puzzle = Singles(SinglesInitState)
singles_puzzle.label = 'Singles Puzzle'

mySearches = [
 #   swiss_puzzle,
    ashgabat_puzzle,
    romania_puzzle,
    switch_puzzle,
    singles_puzzle
]
mySearchMethods = [
]
