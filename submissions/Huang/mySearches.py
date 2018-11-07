import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   Cottontown=dict(Portland=18),
   Fairfield=dict(Mitchellville=21, Portland=17),
   Mitchellville=dict(Portland=7, Fairfield=21),
))

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner'
sumner_puzzle.description = '''
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
romania_map.locations = dict(
    A=( 91, 492), B=(400, 327), C=(253, 288),
    D=(165, 299), E=(562, 293), F=(305, 449),
    G=(375, 270), H=(534, 350), I=(473, 506),
    L=(165, 379), M=(168, 339), N=(406, 537),
    O=(131, 571), P=(320, 368), R=(233, 410),
    S=(207, 457), T=( 94, 410), U=(456, 350),
    V=(509, 444), Z=(108, 531))

romania_puzzle = search.GraphProblem('A', 'B', romania_map)

romania_puzzle.label = 'Romania'
romania_puzzle.description = '''
The simplified map of Romania, per
Russall & Norvig, 3rd Ed., p. 68.


'''

solarSystem_map = search.UndirectedGraph(dict(

    Mars=dict(Earth=53,Venus=187,Satellites=92),
    Earth=dict(Sun=59,Mars=75),
    Venus=dict(Sun=101,Saturn=80),
    Satellites=dict(Mars=118,Jupiter=111),
    Sun=dict(Venus=151,Uranus=75),
    Jupiter=dict(Satellites=111,Neptune=50),
    Neptune=dict(Jupiter=70,Mercury=75),
    Mercury=dict(Neptune=75,Moon=20),
    Saturn=dict(Venus=80,Moon=46,Uranus=97),
    Moon=dict(Uranus=198,Mercury=120),
    Uranus=dict(Saturn=97,Moon=108),
))

solarSystem_puzzle = search.GraphProblem('Earth','Moon', solarSystem_map)
solarSystem_puzzle.label = 'Solar System Map'
solarSystem_puzzle.description = '''
An abbreviated map of Solar System.
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

#initial = 'S'
board = [['B', 'E', 'B', 'E', 'B', 'F'],
        ['B', 'E', 'E', 'E', 'E', 'B'],
        ['R', 'E', 'E', 'E', 'E', 'R'],
        ['B', 'E', 'E', 'R', 'R', 'E'],
        ['E', 'E', 'B', 'E', 'E', 'E'],
        ['S', 'R', 'R', 'E', 'E', 'R']]


class LeftRightMazePuzzle(search.Problem):

    def actions(self, state):
        return ['Up', 'Down', 'Left', 'Right']

    def result(self, state, action, x, y):
        if state == 'E' and action == 'Up':
            return self.result(board[x][y+1], 'Up', x, y+1)
        elif state == 'B' and action == 'Up':
            try:
               return self.result(board[x-1][y], 'Left', x-1, y)
            except:
               return self.result(board[x][y + 1], 'Up', x, y + 1)
        elif state == 'R' and action == 'Up':
            try:
               return self.result(board[x+1][y], 'Right', x+1, y)
            except:
               return self.result(board[x][y + 1], 'Up', x, y + 1)
        elif state == 'E' and action == 'Right':
            return self.result(board[x+1][y], 'Right', x+1, y)
        elif state == 'B' and action == 'Right':
            try:
                return self.result(board[x][y+1], 'Up', x, y+1)
            except:
                return self.result(board[x + 1][y], 'Right', x + 1, y)
        elif state == 'R' and action == 'Right':
            try:
                return self.result(board[x][y-1], 'Down', x, y-1)
            except:
                return self.result(board[x + 1][y], 'Right', x + 1, y)
        elif state == 'E' and action == 'Left':
            return self.result(board[x-1][y], 'Left', x-1, y)
        elif state == 'B' and action == 'Left':
            try:
                return self.result(board[x][y-1], 'Down', x, y-1)
            except:
                return self.result(board[x - 1][y], 'Left', x - 1, y)
        elif state == 'R' and action == 'Left':
            try:
                return self.result(board[x][y+1], 'Up', x, y+1)
            except:
                return self.result(board[x - 1][y], 'Left', x - 1, y)
        elif state == 'E' and action == 'Down':
            return self.result(board[x][y-1], 'Down', x, y-1)
        elif state == 'B' and action == 'Down':
            try:
                return self.result(board[x+1][y], 'Right', x+1, y)
            except:
                return self.result(board[x][y - 1], 'Down', x, y - 1)
        elif state == 'R' and action == 'Down':
            try:
                return self.result(board[x-1][y], 'Left', x-1, y)
            except:
                return self.result(board[x][y - 1], 'Down', x, y - 1)


    def goal_test(self, state):
        return state == 'F'


    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


LRM_puzzle = LeftRightMazePuzzle('S')
LRM_puzzle.label = 'Left Right Maze Puzzle'


mySearches = [
    #   swiss_puzzle,
    #   sumner_puzzle,
    #   romania_puzzle,
    #   switch_puzzle,
    solarSystem_puzzle,
    LRM_puzzle,
]

import random

def flounder(problem, giveup=10000):
    'The worst way to solve a problem'
    node = search.Node(problem.initial)
    count = 0
    while not problem.goal_test(node.state):
        count += 1
        if count >= giveup:
            return null
        children = node.expand(problem)
        node = random.choice(children)
    return node

mySearchMethods = [
    flounder
]

# commented out by whh 2018-10-30
# mySearchMethods = [
#     LRM_puzzle,
#     switch_puzzle,
# ]
