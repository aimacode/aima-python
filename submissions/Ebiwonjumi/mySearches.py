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
#
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
#
# # A trivial Problem definition
# class LightSwitch(search.Problem):
#     def actions(self, state):
#         return ['up', 'down']
#
#     def result(self, state, action):
#         if action == 'up':
#             return 'on'
#         else:
#             return 'off'
#
#     def goal_test(self, state):
#         return state == 'on'
#
#     def h(self, node):
#         state = node.state
#         if self.goal_test(state):
#             return 0
#         else:
#             return 1
#
# #swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
# switch_puzzle = LightSwitch('off')
# switch_puzzle.label = 'Light Switch'
lagos_map = search.UndirectedGraph(dict(
    Lagos=dict(Lekki=41, Ikeja=25, Ikorodu=45, Ikotun=63),
    Lekki=dict(Lagos=41, Epe=58),
    Ikeja=dict(Lagos=25, Ikotun=30, Ijoko=56),
    Ikorodu=dict(Lagos=45, IjebuOde=74, Ijoko=85),
    Epe=dict(Lekki=41, IjebuOde=58),
    Ikotun=dict(Ikeja=30, Lagos=63),
    Ijoko=dict(Ikeja=56, Ikorodu=85),
    IjebuOde=dict(Ikorodu=74, Epe=58),

))

lagos_puzzle = search.GraphProblem('Epe', 'Ikotun', lagos_map)

lagos_puzzle.label = 'Lagos'
lagos_puzzle.description = '''
This is a very simple map of Lagos Nigeria
'''

mySearches = [
    lagos_puzzle,
 #   swiss_puzzle,
 #    romania_puzzle,
 #    switch_puzzle,
]
