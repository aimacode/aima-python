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

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    #sumner_puzzle,
    romania_puzzle,
    switch_puzzle,
    Germany_puzzle,
    Germany_puzzle2
]
mySearchMethods = []