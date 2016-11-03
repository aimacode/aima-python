import search
from math import(cos, pi)

# A sample map problem
zimbabwe_map = search.UndirectedGraph(dict(
    Harare=dict(MountDarwin=118, Kadoma=102, Chikore=138, Mutare=197),
    Kadoma=dict(Harare=102, Gweru=107),
    Gweru=dict(Kadoma=107, Bulawayo=99, Lubimbi=267),
    Bulawayo=dict(Gweru=99, Lubimbi=183),
    Lubimbi=dict(Gweru=267, Bulawayo=183),
    Mutare=dict(Harare=197, Nyanga=98),
    MountDarwin=dict(Harare=118),
    Nyanga=dict(Chikore=106, Mutare=98),
    Chikore=dict(Harare=138, Nyanga=106),
))
zimbabwe_map.locations = dict(
    Harare=(613,200), MountDarwin=(662,101), Chikore=(731,201),
    Nyanga=(769,234), Mutare=(761,311), Kadoma=(512,248),
    Gweru=(502,359), Lumbini=(271,264), Bulawayo=(394,423)
)

harare_mutare_puzzle = search.GraphProblem('Harare', 'Mutare', zimbabwe_map)

harare_mutare_puzzle.label = 'Zimbabwe Map'
harare_mutare_puzzle.description = '''
An abbreviated map of several cities in Zimbabwe.
'''



# A trivial Problem definition
# class HexDeadEnd(search.Problem):
#     def actions(self, state):
#         return ['e', 'se']
#
#     def result(self, state, action):
#         if action == 'e':
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
# initial = set()
# initial = ('11','12','21','22','24','31(S)','32','33','34','35','41','43(F)','44','51',
#               '52','53')
# hex_puzzle = HexDeadEnd(initial)
# hex_puzzle.label = 'Hex Dead End Puzzle'

myPuzzles = [
    harare_mutare_puzzle
    #hex_puzzle,
]