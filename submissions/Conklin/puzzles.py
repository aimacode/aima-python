import search
from math import(cos,pi)

# A sample map problem
sweden_map = search.UndirectedGraph(dict(
    Stockholm=dict(Uppsala=71, Umeå=637, Malmö=271),
    Gothenburg=dict(Örebro=218, Östersund=779, Stockholm=471, Lund=264),
    Malmö=dict(Gothenburg=271, Örebro=502, Helsingborg=64, Lund=19),
    Uppsala=dict(Umeå=571, Stockholm=72),
    Örebro=dict(Stockholm=202, Linköping=122, Umeå=701),
    Linköping=dict(Stockholm=200, Uppsala=265, Norrköping=42, Lund=409),
    Helsingborg=dict(Gothenburg=215, Stockholm=556, Linköping=360),
    Jönköping=dict(Örebro=213, Stockholm=324, Norrköping=167),
    Norrköping=dict(Stockholm=163, Umeå=795, Linköping=42),
    Lund=dict(Stockholm=601, Malmö=19, Umeå=1236),
    Umeå=dict(Östersund=363, Stockholm=639),
    Östersund=dict(Umeå=363, Gothenburg=779, Lund=1156),
))

sweden_map.locations = dict(
    Stockholm=(18.1, 59.3), Gothenburg=(12, 57.7), Malmö=(13, 15.6), Uppsala=(17.6, 59.9), Örebro=(15.2, 59.3),
    Linköping=(15.6, 58.4), Helsingborg=(12.7, 56), Jönköping=(14.2, 57.8), Norrköping=(16.2, 58.6),
    Lund=(13.2, 55.7), Umeå=(20.3, 63.8), Östersund=(14.6, 63.2),
)

#BFS > DFS & UCS > BFS
sweden_puzzle_MU = search.GraphProblem('Malmö', 'Umeå', sweden_map)
#BFS > BeFS
sweden_puzzle_UG = search.GraphProblem('Umeå', 'Gothenburg', sweden_map)
#BeFS > DFS
sweden_puzzle_HJ = search.GraphProblem('Helsingborg', 'Jönköping', sweden_map)
#A* = UCS, but more efficient
sweden_puzzle_UJ = search.GraphProblem('Umeå', 'Jönköping', sweden_map)

#an unnecessary change to commit


sweden_puzzle_MU.label = 'Map of Sweden Using Distance (KM)'
sweden_puzzle_UG.label = 'Map of Sweden Using Approximated Locations'
sweden_puzzle_HJ.label = 'Map of Sweden Using Approximated Locations'
sweden_puzzle_UJ.label = 'Map of Sweden Using Approximated Locations'

#My attempt at a puzzle that, for the life of me, I could not figure out
# class CleaningCrew(search.Problem):
#     def actions(self, state):
#         self.initial = CleaningCrew([
#             ['X', 'X', 'O', 'O', 'O'],
#             ['O', 'X', 'X', 'O', 'O'],
#             ['O', 'X', 'O', 'X', 'O'],
#             ['O', 'O', 'X', 'O', 'X'],
#             ['X', 'O', 'X', 'O', 'O'],
#             ])
#         return ['up', 'down', 'left', 'right']
#
#     def result(self, state, action):
#         if action == 'up':
#             self.columns
#             return 'O'
#         elif action == 'down':
#             self.columns
#             return 'O'
#         elif action == 'right':
#             self.rows
#             return 'O'
#         elif action == 'left':
#             self.rows
#             return 'O'
#         else:
#             return 'X'
#
#     def goal_test(self, state):
#         goal = ([['O', 'O', 'O', 'O', 'O'],
#                  ['O', 'O', 'O', 'O', 'O'],
#                  ['O', 'O', 'O', 'O', 'O'],
#                  ['O', 'O', 'O', 'O', 'O'],
#                  ['O', 'O', 'O', 'O', 'O'],
#                  ])
#         return state == goal
#
#     def h(self, node):
#         state = node.state
#         if self.goal_test(state):
#             return 0
#         else:
#             return 1
#
#     CleaningCrewPuzzle = CleaningCrew(initial)
#     CleaningCrewPuzzle.label = 'A Simplified Cube Puzzle'

myPuzzles = [
    sweden_puzzle_MU,
    sweden_puzzle_UG,
    sweden_puzzle_HJ,
    sweden_puzzle_UJ,
    #CleaningCrewPuzzle,
]