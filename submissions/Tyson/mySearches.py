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
from utils import is_in

potter_map = search.UndirectedGraph(dict(
    Amarillo=dict(Washburn=15, Panhandle=34),
    Canyon=dict(Umbarger=10, Happy=22, VigoPark=35),
    Washburn=dict(Amarillo=15, Claude=14),
    Umbarger=dict(Canyon=10, Arney=15),
    Arney=dict(Umbarger=15, Nazareth=15),
    Nazareth=dict(Arney=15, Happy=20, Tulia=22, Dimmit=12),
    Happy=dict(Nazareth=20, Canyon=22, Tulia=18),
    Tulia=dict(Nazareth=22, Happy=18, Silverton=30, VigoPark=20),
    Panhandle=dict(Claude=20, Fritch=25, Amarillo=34),
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
    Dimmit=(-12, 0), VigoPark=(40, 18), BoysRanch=(0, 100), Masterson=(30, 100),
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
potter_puzzle2.label = 'Potter County - Arney to BoysRanch'
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


# ColorPuzzle_3x3Grid = search.UndirectedGraph(dict(
#     One=dict(Two='Right', Four="Down"),
#     Two=dict(One='Left', Three="Right", Five="Down"),
#     Three=dict(Two="Left", Six="Down"),
#     Four=dict(Five="Right", Seven="Down", One="Up"),
#     Five=dict(Four='Left', Six="Right", Two="Up", Eight="Down"),
#     Six=dict(Nine="Finish"),
#     Seven=dict(Four="Up", Eight="Right"),
#     Eight=dict(Nine="Finish")
#
# ))

HousePuzzle_Map = dict(
    a1=dict(a2=1, b1=1),
    a2=dict(a1=1, b2=1, a3='tree'),
    a3=dict(a3='tree'),
    a4=dict(b4=1),
    b1=dict(c1=1, b2=1),
    b2=dict(b1=1, b3='mud', a2=1, c2='tree'),
    b3=dict(b3='mud', a3='tree', b4=1, c3='tree'),
    b4=dict(a4=1, c4=1),
    c1=dict(d1=1, c2='tree', b1=1),
    c2=dict(c2='tree'),
    c3=dict(c3='tree'),
    c4=dict(c3='tree', b4=1, d4=1),
    d1=dict(c1=1, d2=1),
    d2=dict(c2='tree', d3=1),
    d3=dict(d4=1, d2=1),
    d4=dict(d3=1, c4=1),


)

HousePuzzle_MapGridLocations = dict(
    a1=(1, 1), a2=(1, 2), a3=(1, 3), a4=(1, 4),
    b1=(2, 1), b2=(2, 2), b3=(2, 3), b4=(2, 4),
    c1=(3, 1), c2=(3, 2), c3=(3, 3), c4=(3, 4),
    d1=(4, 1), d2=(4, 2), d3=(4, 3), d4=(4, 4),
)

HousePuzzle_MapHouseLocations = dict(
    a1="house1",
    d4="house2"
)





#HousePuzzle_Map.= dict()


class HousePuzzle(search.Problem):

    def __init__(self, map, locations, houseLocations, start, finish):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = start
        self.finish = finish
        self.map = map
        self.locations = locations
        self.houseLocations = houseLocations


    def actions(self, state):
        neighbors = self.map[state]
        openSpaces = []
        for x in neighbors:
            if neighbors.get(x) != 'tree' and neighbors.get(x) != 'mud':
                openSpaces.append(x)

            elif neighbors.get(x) == 'mud':
                neighbors.update({x: 3})
                openSpaces.append(x)

            else:
                continue

        return openSpaces

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.finish

    def path_cost(self, c, state1, action, state2):
        neighbors = self.map[state1]
        cost = neighbors[state2]
        return c + cost

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


#switch_puzzle = search.GraphProblem('Off', 'On', potter_map)
# switch_puzzle = LightSwitch('off')
# switch_puzzle.label = 'Light Switch'

house_puzzle = HousePuzzle(HousePuzzle_Map, HousePuzzle_MapGridLocations, HousePuzzle_MapHouseLocations, "a1", "d4")
house_puzzle.label = 'House Maze'


mySearches = [
 #   swiss_puzzle,
 #    potter_puzzle,
    house_puzzle,
    potter_puzzle2,
    # potter_puzzle3
    #romania_puzzle,
    #switch_puzzle,
]

mySearchMethods = []
