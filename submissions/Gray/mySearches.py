import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))

gray_map = search.UndirectedGraph(dict(
   Boston=dict(Cambridge=14, Brookline=21, Chelsea=13, Arlington=3),
   Chelsea=dict(Boston=13, Winthrop=13, Somerville=16, Arlington=8),
   Winthrop=dict(Chelsea=13),
   Somerville=dict(Chelsea=16, Cambridge=10, Medford=5, Arlington=6),
   Cambridge=dict(Boston=14, Brookline=21, Watertown=16, Belmont=15,
                  Medford=15, Somerville=10, Arlington=10),
   Medford=dict(Cambridge=15, Belmont=5, Somerville=5),
   Brookline=dict(Boston=21, Cambridge=21, Watertown=20),
   Watertown=dict(Belmont=7, Cambridge=16, Brookline=20),
   Belmont=dict(Watertown=7, Cambridge=15, Medford=5),
   Arlington=dict(Somerville=6, Cambridge=10, Boston=3, Chelsea=8)
))

gray_puzzle = search.GraphProblem('Belmont', 'Chelsea', gray_map)

gray_puzzle.label = 'Boston'
gray_puzzle.description = '''
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
    gray_puzzle,
    romania_puzzle,
    switch_puzzle,
]

mySearchMethods = []