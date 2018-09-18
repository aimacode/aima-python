import search
from math import(cos, pi)


# A sample map problem
sumner_map = search.UndirectedGraph(dict(
   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   Cottontown=dict(Portland=18),
   Fairfield=dict(Mitchellville=21, Portland=17),
   Mitchellville=dict(Portland=7, Fairfield=21),
))

# Nashville, Atlanta, College Station, Baltimore, Raleigh, St. Louis, Gainsville,

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner'
sumner_puzzle.description = '''
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


la_map = search.UndirectedGraph({
    'Beverly Hills': {'Hollywood': 15, 'Santa Monica': 15},
    'Calabasas': {'Westlake Village': 15, 'Woodland Hills': 11},
    'Disneyland': {'Downtown': 28, 'Venice Beach': 51},
    'Downtown': {'Disneyland': 28, 'Hollywood': 16, 'Santa Monica': 20},
    'Hollywood': {'Beverly Hills': 15, 'Downtown': 16, 'Woodland Hills': 24},
    'Malibu': {'Santa Monica': 33, 'Westlake Village': 26},
    'Santa Monica': {'Beverly Hills': 15, 'Downtown': 20, 'Malibu': 33, 'Venice Beach': 8},
    'Venice Beach': {'Disneyland': 51, 'Santa Monica': 8, 'Woodland Hills': 32},
    'Westlake Village': {'Calabasas': 15, 'Malibu': 26},
    'Woodland Hills': {'Calabasas': 11, 'Hollywood': 24, 'Venice Beach': 32}
})

la_puzzle = search.GraphProblem('Woodland Hills', 'Disneyland', la_map)
la_puzzle.label = 'Los Angeles'

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
 #   sumner_puzzle,
 #  romania_puzzle,
 #   switch_puzzle,
    la_puzzle
]

mySearchMethods = []
