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

wayne_map.label = 'Sumner'
wayne_map.description = '''
An abbreviated map of Sumner County, TN.
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

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    myPuzzle,
    romania_puzzle,
    switch_puzzle,
]

mySearchMethods = []
