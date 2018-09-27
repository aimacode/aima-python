
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
  SpringCreek=dict(ThreeWay=18, Medon=34, Humboldt=29)
))

# Coordinates for map. May not be entirely accurate but as close as possible
madison_map.locations = (dict(
   Jackson=(485, 512),
   Humboldt=(482, 482),
   ThreeWay=(474, 474),
   Medon=(495, 501),
   SpringCreek=(474, 464)))

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
# one entrance and one exit. The costs right now may seem a bit random due to time constraint
# but, if I return back to this project for SURS, I'll try to make it more reasonable.

# Also, instead of using a 2D array to create a 2D maze. I decided to use a dictionary of dictionaries to make
# maze/labyrinth of points that connect to each other. So really its more of a path maze rather than a traditional maze.

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
    M=dict(N=43, L=6),
    N=dict(O=64, M=43),
    O=dict(W=80, P=12, N=64),
    P=dict(Q=12, O=20),
    Q=dict(C=20, U=20, R=45, P=12),
    R=dict(T=62, Q=45),
    T=dict(AJ=32, R=62),
    U=dict(V=96, Q=20),
    V=dict(W=52, AF=20, U=96),
    AF=dict(AE=51, V=20),
    AE=dict(AD=46, AF=51),
    AD=dict(AG=12, AE=46),
    AG=dict(AH=52, AM=46, AD=12),
    AH=dict(AI=21, AG=52),
    AI=dict(AJ=21, AH=21),
    AJ=dict(T=32, AI=21),
    AM=dict(Finish=65, AG=46),
    W=dict(V=52, X=23, O=80),
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
# However, labyrinths may take longer to solve and give more interesting outputs due to the amount of paths.

class Maze2(search.Problem):

    def __init__(self, initial, goal, maze):
        self.maze = maze
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        bob = self.maze[state]
        keys = bob.keys()
        return keys

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        bob = self.maze[state1]
        cost = bob[state2]
        return c + cost

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


# Problem defintion for the Map coordinates. I could have combined it with the one above, but there were plenty of errors
# because of the added location attribute.
from grid import distance

class Map4(search.Problem):

    def __init__(self, initial, goal, map2, location):
        self.map2 = map2
        self.location = location
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        bob = self.map2[state]
        keys = bob.keys()
        return keys

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        bob = self.map2[state1]
        cost = bob[state2]
        return c + cost

    def h(self, node):
        state = node.state
        coor1 = self.location[state]
        coor2 = self.location[self.goal]
        return distance(coor1,coor2)

   # def h(self, node):
      #  state = node.action
       # state1 = self.initial
        #state2 = self.map


maze_puzzle2 = Maze2('Start', 'Finish', maze_path)

maze_puzzle2.label = 'Maze'

Labyrinth_puzzle = Maze2('Start','Finish', Labyrinth_path)

Labyrinth_puzzle.label = 'Labyrinth'

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)

switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

# Puzzle using coordinates
madison_puzzle4 = Map4('SpringCreek','Jackson', madison_map.dict, madison_map.locations)

madison_puzzle4.label = 'Madison1 w/ Coordinates'
madison_puzzle4.description = 'Coordinates'

mySearches = [
    madison_puzzle,
   # romania_puzzle,
   #  switch_puzzle,
    madison_puzzle1,
    madison_puzzle4,
    maze_puzzle2,
    Labyrinth_puzzle
]


#def The_Shining(problem):
 #   node = search.Node(problem.initial)
  #  count = 0
   # while not problem.goal_test(node.state):
    #  for child in node.expand(problem):
     #     count += 1
      #    bob8=child
       #   currentnode = child.expand(problem)
        #  if count == 50:
         #    return currentnode.state
          #if problem.goal_test(child.state):
           #   return child

mySearchMethods = [
   #The_Shining(maze_puzzle2)
]