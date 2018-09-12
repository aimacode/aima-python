import search
from math import (cos, pi)

# A sample map problem

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
#
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''
# romania_map = search.UndirectedGraph(dict(
#     A=dict(Z=75,S=140,T=118),
#     Z=dict(O=71,A=75),
#     S=dict(O=151,R=80,F=99),
#     T=dict(A=118,L=111),
#     O=dict(Z=71,S=151),
#     L=dict(T=111,M=70),
#     M=dict(L=70,D=75),
#     D=dict(M=75,C=120),
#     R=dict(S=80,C=146,P=97),
#     C=dict(R=146,P=138,D=120),
#     F=dict(S=99,B=211),
#     P=dict(R=97,C=138,B=101),
#     B=dict(G=90,P=101,F=211),
# ))
#
# romania_puzzle = search.GraphProblem('A', 'B', romania_map)
#
# romania_puzzle.label = 'Romania'
# romania_puzzle.description = '''
# The simplified map of Romania, per
# Russall & Norvig, 3rd Ed., p. 68.
# '''

cotes_dazure_map = search.UndirectedGraph(dict(
    Marseille=dict(Manosque=65, Aubagne=25),
    Aubagne=dict(Marseille=25, Toulon=34, Brignoles=46),
    Toulon=dict(SaintTropez=71, Brignoles=41, Aubagne=34),
    Brignoles=dict(Aubagne=34, Toulon=41, Frejus=46, Draguignan=56),
    Manosque=dict(Marseille=65, Gap=74),
    SaintTropez=dict(Toulon=71, Frejus=49),
    Frejus=dict(SaintTropez=49, Brignoles=46, Nice=53),
    Draguignan=dict(Brignoles=56, Digne=113),
    Gap=dict(Manosque=74, Digne=93),
    Digne=dict(Draguignan=113, Gap=93, Nice=146),
    Nice=dict(Frejus=53, Digne=146)
))

cotes_dazure_puzzle = search.GraphProblem('Marseille', 'Nice', cotes_dazure_map)

cotes_dazure_puzzle.label = "Cotes d'Azur"

cotes_dazure_puzzle.description = '''
An abbreviated map of Provence-Alpes-Côte d’Azur, France.
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


# swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
    #   swiss_puzzle,
    # sumner_puzzle,
    # romania_puzzle,
    cotes_dazure_puzzle,
    switch_puzzle,

]
