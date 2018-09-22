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


# class FloodPuzzle(search.Problem):
#
#     def __init__(self, initial, goal, board):
#
#         self.initial = initial
#         self.goal = goal
#         self.board = board
#         self.state = initial
#
#     def actions(self, state):
#         return ['left', 'right', 'up', 'down']
#
#     def result(self, state, action):
#         return
#
#     def goal_test(self, state):
#         return self.state == self.goal
#
#
# flood_game_board = dict(
#     A1=dict(right=2, down=2), A2=dict(left=1, right=3, down=1),
#     A3=dict(left=2, down=3), B1=dict(right=1, down=1, up=1),
#     B2=dict(left=2, right=3, down=3, up=2), B3=dict(left=1, down=2, up=3),
#     C1=dict(up=2, right=3), C2=dict(up=1, left=1, right=2), C3=dict(left=3, up=3))
#
# initial_board = [['A1', 1], ['A2', 2], ['A3', 3],
#                  ['B1', 2], ['B2', 1], ['B3', 3],
#                  ['C1', 1], ['C2', 3], ['C3', 2]]
# goal_board = [['A1', 1], ['A2', 1], ['A3', 1],
#               ['B1', 1], ['B2', 1], ['B3', 1],
#               ['C1', 1], ['C2', 1], ['C3', 1]]
# flood_puzzle = FloodPuzzle(initial_board, goal_board, flood_game_board)
# flood_puzzle.label = 'Flood Puzzle'


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
    #flood_puzzle,
]

mySearchMethods = []
