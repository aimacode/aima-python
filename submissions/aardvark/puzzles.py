import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
 #   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
 #   Cottontown=dict(Portland=18),
 #   Fairfield=dict(Mitchellville=21, Portland=17),
 #   Mitchellville=dict(Portland=7, Fairfield=21),


 #   A=dict(D=70, C=80, B=140),
 #   D=dict(A=70,C=100,Z=200),
 #   C=dict(A=80,D=100,E=80,B=70),
 #   B=dict(A=140,C=70,E=90,Z=130),
 #   E=dict(C=80,B=90,Z=60),
 #   Z=dict(D=200,E=60,B=130),


#    A=dict(B=70,C=80,E=100),
#    B=dict(A=70),
#    C=dict(A=80,E=100,D=60),
#    E=dict(A=100,C=100,Z=150),
#    D=dict(C=60,Z=90),
#    Z=dict(D=90,E=150),


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

#sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
sumner_puzzle = search.GraphProblem('A', 'B', sumner_map)

sumner_puzzle.label = 'Sumner Map'
sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
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

myPuzzles = [
 #   swiss_puzzle,
    sumner_puzzle,
    switch_puzzle,
]