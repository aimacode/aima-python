import search
from math import(cos, pi)

# A sample map problem
orange_map = search.UndirectedGraph(dict(
   PortJervis=dict(Middletown=28, Warwick=37, Monticello=30),
   Monticello=dict(PortJervis=30),
   Middletown=dict(PortJervis=28, Warwick=29, Newburgh=32, Wallkill=33, Washingtonville=12),
   Warwick=dict(PortJervis=37, GreenwoodLake=11, Newburgh=44),
   Newburgh=dict(Warwick=44, Middletown=32, Poughkeepsie=30, Wallkill=32, Washingtonville=10),
   Poughkeepsie=dict(Newburgh=30),
   Wallkill=dict(Middletown=33, Newburgh=32),
   Washingtonville=dict(Warwick=25, Newburgh=10, Middletown=12),
))

orange_puzzle = search.GraphProblem('PortJervis', 'Newburgh', orange_map)

orange_puzzle.label = 'Orange'
orange_puzzle.description = '''
An abbreviated map of Orange County, NY.
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
    orange_puzzle,
    romania_puzzle,
    switch_puzzle,
]
