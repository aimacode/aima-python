
import search
import numpy as np
from math import(cos, pi)

# A sample map problem
from utils import is_in

madison_map = search.UndirectedGraph(dict(
 # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
  # Cottontown=dict(Portland=18),
   #Fairfield=dict(Mitchellville=21, Portland=17),
   #Mitchellville=dict(Portland=7, Fairfield=21),
  Jackson=dict(Humboldt=27),
  Humboldt=dict(Jackson=27, ThreeWay=8),
  ThreeWay=dict(Humboldt=8, Medon=34),
  Medon=dict(Jackson=17, Humboldt=43,ThreeWay=34),
  SpringCreek=dict(ThreeWay=18 , Medon=34 , Humboldt=29)
))

madison_puzzle = search.GraphProblem('Jackson', 'ThreeWay', madison_map)
madison_puzzle1 = search.GraphProblem('SpringCreek', 'Jackson', madison_map)

madison_puzzle.label = 'Madison'
madison_puzzle.description = '''
An abbreviated map of Madison County, TN.
This map is unique, to the best of my knowledge.
'''
madison_puzzle1.label = 'Madison1'
madison_puzzle1.description = '''
An abbreviated map of Madison County, TN.
This map is unique, to the best of my knowledge.
'''

madison_map.locations = dict(
    Jackson=(482, 512), Humboldt=(482, 482), ThreeWay=(474, 474), Medon=(495, 501), SpringCreek=(474, 464))

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
# 0s Represent Walls
# 1s Represent Path
# 9 Represents Start
# 8 Represents Exit

Labyrinth2 = np.array([[9, 1, 1, 1, 1, 1],
                       [0, 1, 0, 1, 1, 1],
                       [0, 1, 0, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1, 1],
                       [8, 1, 1, 1, 1, 1]])

# Above is the visual representation of the below labyrinth. It has multiple paths to traverse but only
# one entrance and one exit.

Labyrinth_path = (dict(
    Start=dict(B=2),
    B=dict(C=2, Start=2),
    C=dict(D=5, Q=20, B=2),
    D=dict(E=8, C=5),
    E=dict(F=14, D=8),
    F=dict(G=13, E=14),
    G=dict(H=24, F=13),
    H=dict(I=34, G=24),
    I=dict(J=78, H=34),
    J=dict(AW=54, I=78),
    AW=dict(K=56, AC=21, J=54),
    K=dict(L=87, AW=56),
    L=dict(M=6, K=87),
    M=dict(N=75, L=86),
    N=dict(O=64, M=43),
    O=dict(W=42, P=52, N=64),
    P=dict(Q=12, O=52),
    Q=dict(C=20, U=20, R=45, P=31),
    R=dict(T=51, Q=96),
    T=dict(U=5, AJ=54, R=62),
    U=dict(V=52, T=31, Q=52),
    V=dict(W=85, AF=20, U=96),
    AF=dict(AE=85, V=12),
    AE=dict(AD=51, AF=51),
    AD=dict(AG=12, N=95, AE=46),
    AG=dict(AH=73, AM=46, AD=20),
    AH=dict(AI=21, AG=52),
    AI=dict(AJ=51, AH=21),
    AJ=dict(T=32, AI=21),
    AM=dict(Finish=65, AG=75),
    W=dict(V=52, X=23, O=42),
    X=dict(Y=56, W=23),
    Y=dict(Z=12, X=56),
    Z=dict(AB=21, Y=12),
    AB=dict(AC=12, Z=21),
    AC=dict(AB=12, AW=21),
    Finish=dict(AM=65),
))


Maze = np.array([[0,0,9,0,0],
                 [1,1,1,1,1],
                 [1,0,1,0,1],
                 [1,0,0,0,1],
                 [1,1,8,0,0]])

# Above is a visual representation of the below maze. Unlike the labyrinth, it has only one solution
# to get from the start to finish.

maze_path = (dict(
    Start=dict(A=0),
    A=dict(Start=0, C=0, B=0, F=0),
    B=dict(A=0, G=0),
    C=dict(A=0, D=0),
    D=dict(M=0, C=0),
    M=dict(L=0, D=0),
    L=dict(K=0, M=0),
    K=dict(L=0, Finish=0),
    Finish=dict(K=0),
    G=dict(B=0, H=0),
    H=dict(G=0, J=0),
    J=dict(I=0, H=0),
    I=dict(J=0),
    F=dict(A=0, O=0),
    O=dict(F=0, N=0),
    N=dict(O=0, Q=0),
    Q=dict(N=0,P=0),
    P=dict(Q=0),
))

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

# This problem definition solves any size maze and labyrinth if given enough memory space.
# However, labyrinths take longer to solve due to the amount of paths.

class Maze2(search.Problem):

    def __init__(self, initial, goal, map):
        self.map = map
        self.initial = initial
        self.goal = goal
        self.maze = map

    def actions(self, state):
        bob = self.map[state]
        keys = bob.keys()
        return keys

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        bob = self.map[state1]
        cost = bob[state2]
        return c + cost

    #def value(self, state):

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

maze_puzzle2 = Maze2('Start', 'Finish', maze_path)

maze_puzzle2.label = 'Maze'

Labyrinth_puzzle = Maze2('Start','Finish', Labyrinth_path)

Labyrinth_puzzle.label = 'Labyrinth'

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'


mySearches = [
    madison_puzzle,
   # romania_puzzle,
   #  switch_puzzle,
    madison_puzzle1,
    maze_puzzle2,
    #Labyrinth_puzzle
]

import array


def The_Shining(problem, Getcaught = 9000):
    footsteps = []
    node = search.Node(problem.initial)
    count =0
    while not problem.goal_test(node):
        count+=1
        footsteps.append(node.action)
        node.expand(problem.initial)
        if problem.goal_test(node):
            node.solution()



    print(footsteps)

mySearchMethods = [
   The_Shining(maze_puzzle2)
]