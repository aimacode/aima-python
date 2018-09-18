
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
#0s Represent Walls
#1s Represent Path
maze2 = np.array([[9, 1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 1, 1],
                  [0, 1, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1],
                  [4, 6, 2, 0, 0, 1],
                  [6, 3, 2, 1, 1, 0],
                  [8, 5, 3, 1, 1, 1]])

maze2_path = search.UndirectedGraph(dict(
    Start=dict(B=2),
    B=dict(C=2, Start=2),
    C=dict(D=5, Q=5, B=5),
    D=dict(E=8, C=9),
    E=dict(F=10, D=12),
    F=dict(G=13, E=14),
    G=dict(H=24, F=12),
    H=dict(I=34, G=56),
    I=dict(J=34, H=67),
    J=dict(AW=54, I=78),
    AW=dict(K=56, AC=21, J=34),
    K=dict(L=78, AW=89),
    L=dict(AX=56, M=76, K=87),
    AX=dict(L=43),
    M=dict(N=75, L=86),
    N=dict(O=64, M=43),
    O=dict(W=42, P=71, N=90),
    P=dict(Q=12, O=52),
    Q=dict(C=98, U=20, R=45, P=31),
    R=dict(T=51, S=51, Q=96),
    S=dict(R=85),
    T=dict(U=5, AJ=54, R=62),
    U=dict(V=52, T=31, Q=52),
    V=dict(W=85, AF=20, U=96),
    AF=dict(AE=85, V=12),
    AE=dict(AD=51, AF=51),
    AD=dict(AG=12, N=95, AE=46),
    AG=dict(AH=73, AM=46, AD=20),
    AH=dict(AI=21, AG=52),
    AI=dict(AJ=51, AH=21),
    AJ=dict(T=32, AI=21, AK=75),
    AK=dict(AL=46, AJ=85),
    AL=dict(AK=26),
    AM=dict(Finish=65, AG=75),
    W=dict(V=52, X=23, O=12),
    X=dict(Y=56, W=45),
    Y=dict(Z=12, X=91),
    Z=dict(AB=21, Y=51),
    AB=dict(AC=12, Z=82),
    Finish=dict(AM=96),
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


class Maze2(search.Problem):

    def __init__(self, initial, goal, maze):
        self.initial = initial
        self.goal = goal
        self.maze = maze

    def actions(self, state):
        bob = self.maze[state]
        key = bob.keys()
        return key

    def result(self, state, action):
        return action

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

   #def path_cost(self, c, state1, action, state2):


    #def value(self, state):

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1




maze_puzzle2 = Maze2('Start', 'Finish', maze2_path)

maze_puzzle2.label = 'Maze'



#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'


mySearches = [
    madison_puzzle,
   # romania_puzzle,
  #  switch_puzzle,
    madison_puzzle1,
    maze_puzzle2,



]
mySearchMethods = []