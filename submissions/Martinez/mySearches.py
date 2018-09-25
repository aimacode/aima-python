import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))

# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
#
# sumner_puzzle.label = 'Sumner'
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''


# My map
Germany_map = search.UndirectedGraph(dict(
    Bremen=dict(Hamburg=81, Hanover=93, Dortmund=157),
    Hamburg=dict(Bremen=81, Hanover=119, Berlin=175),
    Hanover=dict(Bremen=93, Hamburg=119, Berlin=189, Dortmund=164, Leipzig=171, Frankfurt=221),
    Berlin=dict(Hamburg=175, Hanover=189, Leipzig=137, Dresden=136),
    Leipzig=dict(Berlin=137, Hanover=171, Nuremberg=170, Dresden=82),
    Dresden=dict(Berlin=136, Leipzig=82, Nuremberg=210),
    Dortmund=dict(Bremen=157, Hanover=164, Frankfurt=149, Essen=31),
    Essen=dict(Dortmund=31, Dusseldorf=34),
    Dusseldorf=dict(Essen=34, Cologne=40),
    Cologne=dict(Dusseldorf=40, Frankfurt=120),
    Frankfurt=dict(Cologne=120, Hanover=221, Nuremberg=152, Stuttgart=129),
    Nuremberg=dict(Frankfurt=152, Leipzig=170, Stuttgart=132, Munich=101),
    Stuttgart=dict(Frankfurt=129, Nuremberg=132, Munich=136),
    Munich=dict(Stuttgart=136, Nuremberg=101)

))
Germany_map.locations = dict(
    Bremen=(53.07, 8.8), Hamburg=(53.55, 9.99), Hanover=(52.37, 9.73), Berlin=(52.52, 13.45), Leipzig=(51.33, 12.37),
    Dresden=(51.05, 13.73), Dortmund=(51.51, 7.46), Essen=(51.45, 7.01), Dusseldorf=(51.22, 6.77), Cologne=(50.93, 6.96)
    , Frankfurt=(50.11, 8.68), Nuremberg=(49.45, 11.07), Stuttgart=(48.77, 9.18), Munich=(48.13, 11.58)
)


Germany_puzzle = search.GraphProblem('Bremen', 'Munich', Germany_map)
Germany_puzzle.label = 'Bremen to Munich'
Germany_puzzle.description = 'go from Bremen to Munich...if you can'
Germany_puzzle2 = search.GraphProblem('Munich', 'Essen', Germany_map)
Germany_puzzle2.label = 'Munich to Essen'
Germany_puzzle2.description = 'the most difficult one so far.'


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

#my puzzle
# Dungeonpuzzle_map = dict(
#         E1=dict(E2='room'),
#         E2=dict(D2=0, F2=0),
#         F2=dict(E2=0),
#         D2=dict(C2='intersection'),
#         C2=dict(B2='nada', C3=1),
#         B2=dict(A2='Dead end'),
#         A2=dict(B2='nada'),
#         C3=dict(C4='intersection 2'),
#         C4=dict(B4='exit', D4='dead end 2'),
#         B4=dict(C4='intersection 2'),
#         D4=dict(C4='intersection 2'))
# #start at B
# #exit is at D
# DungeonGrid = [['E'], ['H'], ['E'], ['E'],
#                ['E'], ['H'], ['E'], ['D'],
#                ['E'], ['H'], ['H'], ['H'],
#                ['E'], ['H'], ['E'], ['H'],
#                ['B'], ['H'], ['E'], ['E'],
#                ['E'], ['H'], ['E'], ['E']]
#
# class Dungeonpuzzle(search.Problem):
#     def __init__(self,start, end):
#         self.map = map
#         self.initial = start
#         self.goal = end
#
#     def actions(self , state):
#
#             nearCells = self.map[state]
#             keys = nearCells.keys()
#             return keys
#
#     def result(self, state, action):
#
#         return action
#
#     def goal_test(self, state):
#         return state == self.goal
#
#     def path_cost(self, c, state1, action, state2):
#         nearCells = self.map[state1]
#         cost = nearCells[state2]
#         return c + cost
#
#
# dungeon1 = Dungeonpuzzle('room', 'exit', Dungeonpuzzle_map)
# dungeon1.label = 'dungeon'



#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    #sumner_puzzle,
    romania_puzzle,
    switch_puzzle,
    Germany_puzzle,
    Germany_puzzle2,
    #dungeon1
]
mySearchMethods = []