import search
from math import(cos, pi)

# A sample map problem
#sumner_map = search.UndirectedGraph(dict(
   # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   # Cottontown=dict(Portland=18),
   # Fairfield=dict(Mitchellville=21, Portland=17),
   # Mitchellville=dict(Portland=7, Fairfield=21),

#))

#My map problem
potter_map = search.UndirectedGraph(dict(
    Amarillo=dict( Washburn=15),
    Canyon=dict( Umbarger=10, Happy=22, VigoPark=35),
    Washburn=dict(Amarillo=15, Claude=14),
    Umbarger=dict(Canyon=10, Arney=15),
    Arney=dict(Umbarger=15, Nazareth=15),
    Nazareth=dict(Arney=15, Happy=20, Tulia=22, Dimmit=12),
    Happy=dict(Nazareth=20, Canyon=22, Tulia=18),
    Tulia=dict(Nazareth=22, Happy=18, Silverton=30, VigoPark=20),
    Panhandle=dict( Claude=20, Fritch=25),
    Claude=dict(Washburn=14, Panhandle=20),
    Silverton=dict(Tulia=30, VigoPark=20),
    Dimmit=dict(Nazareth=12),
    VigoPark=dict(Tulia=20, Silverton=30, Happy=28, Claude=35),
    Masterson=dict(Amarillo=31, BoysRanch=30),
    Fritch=dict(Masterson=15, Panhandle=25),
    Groom=dict(Claude=10, Panhandle=10),
    Love=dict(Fritch=29, Groom=7),





    ))

potter_map.locations = dict(
    Amarillo=(20, 16), Canyon=(10, 35), Washburn=(35, 65), Umbarger=(0, 30), Arney=(0, 15),
    Nazareth=(1, 0), Happy=(7, 12), Tulia=(22, 0), Panhandle=(50, 80), Claude=(52, 60), Silverton=(52, 0),
    Dimmit=(-12, 0), VigoPark=(40, 18), BoysRanch=(0,100), Masterson=(30, 100),
    Fritch=(32, 75), Groom=(51, 70), Love=(42, 75),

)


#sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

#sumner_puzzle.label = 'Sumner'
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''
# potter_puzzle = search.GraphProblem('Canyon', 'Nazareth', potter_map)
# potter_puzzle.label = 'Potter County - Canyon to Nazareth'
# potter_puzzle.description = '''First Instance '''

potter_puzzle2 = search.GraphProblem('Arney', 'BoysRanch', potter_map)
potter_puzzle2.label = 'Potter County - Claude to Silverton'
potter_puzzle2.description = '''Instance where BFS does better than DFS '''

# potter_puzzle3 = search.GraphProblem('Nazareth', 'Amarillo', potter_map)
# potter_puzzle3.label = 'Potter County-Nazareth to Washburn'
# potter_puzzle3.description = '''Instance where UniformCost does better than DFS and BFS '''




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


# ColorPuzzle_Grid = search.UndirectedGraph(dict(
#
#
# ))

# class ColorPuzzle(search.Problem):
#
#     def actions(self,state):
#         return['up','down','left','right']
#
#     def result(self, state, action):


#switch_puzzle = search.GraphProblem('Off', 'On', potter_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
 #    potter_puzzle,
    potter_puzzle2,
    # potter_puzzle3
    #romania_puzzle,
    #switch_puzzle,
]

mySearchMethods = []
