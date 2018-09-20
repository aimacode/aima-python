import search
import array
import random
from math import(cos, pi)

franklin_map = search.UndirectedGraph(dict(
    Hilliard=dict(UpperArlington=18, Valleyview=11, Dublin=15),
    GroveCity=dict(CanalWinchester=33, Valleyview=21, Obetz=18),
    UpperArlington=dict(Columbus=14, Worthington=17),
    Columbus=dict(Valleyview=11, CanalWinchester=27),
    Worthington=dict(NewAlbany=21, Gahanna=22),
    NewAlbany=dict(CanalWinchester=35, Gahanna=15),
    Gahanna=dict(Bexley=20, Reynoldsburg=13),
    Bexley=dict(Columbus=14, Renyoldsburg=28),
    CanalWinchester=dict(Reynoldsburg=27, Obetz=17),
    Valleyview=dict(Obetz=23)
))
franklin_puzzle = search.GraphProblem('Hilliard', 'CanalWinchester', franklin_map)

franklin_puzzle.label = 'Franklin'
franklin_puzzle.description = '''
An abbreviated map of Franklin County, OH.
This map is unique, to the best of my knowledge.
'''

ohio_map = search.UndirectedGraph(dict(
    Kenton=dict(Ottawa=55, Troy=77,Mansfield=70, Urbana=53),
    Troy=dict(Dayton=28, Lima=54, Kenton= 75),
    London=dict(Dayton=60, Chillicothe=66, Urbana=35),
    Chillicothe=dict(Lebanon=77, Hillsboro=52),
    Columbus=dict(Athens=81, Mansfield=66, Urbana=51),
    Akron=dict(Mansfield=63, Cleveland=53, NewPhil=54),
    NewPhil=dict(Newark=79, Woodsfield=92),
    Lebanon=dict(Dayton=35, London=60, Hillsboro=56),
))
ohio_puzzle = search.GraphProblem('Cleveland', 'Troy', ohio_map)
ohio_puzzle.label = 'Ohio'


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


def grid_initial(grid_dimensions):
    grid = {}
    for x in range(grid_dimensions):
        for y in range(grid_dimensions):
            if y == 0:
                if x == 2:
                    grid[x, y] = 'T'
                else:
                    grid[x, y] = '.'
            elif y == 1:
                if x == 3:
                    grid[x, y] = 'T'
                else:
                    grid[x, y] = '.'
            elif y == 2:
                grid[x, y] = '.'
            elif y == 3:
                if x == 0:
                    grid[x, y] = 'T'
                else:
                    grid[x, y] = '.'
        print(grid)
        return grid


def solved_grid(grid_dimensions):
    grid = {}
    for x in range(grid_dimensions):
        for y in range(grid_dimensions):
            if y == 0:
                if x == 1:
                    grid[x, y] = 't'
                elif x == 2:
                    grid[x, y] = 'T'
                else:
                    grid[x, y] = '.'
            elif y == 1:
                if x == 3:
                    grid[x, y] = 'T'
                else:
                    grid[x, y] = '.'
            elif y == 2:
                if x == 3:
                    grid[x, y] = 't'
                else:
                    grid[x, y] = '.'
            elif y == 3:
                if x == 0:
                    grid[x, y] = 'T'
                elif x == 1:
                    grid[x, y] = 't'
                else:
                    grid[x, y] = '.'
        print(grid)
        return grid


class Tents(search.Problem):
    def __init__(self, initial_state, goal_state, start_x, start_y):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.current_state = initial_state
        self.start_x = start_x
        self.start_y = start_y

    def actions(self, state):
        return ['t', '.']

    def result(self, state, action):
        updated_state = self.current_state
        # t = tent, T = tree, . = blank
        for x in range(4):
            for y in range(4):
                if action == 't':
                    if self.current_state[x, y] == '.':
                        self.current_state[x, y] ='t'
                elif self.current_state[x, y] == 'T':
                    return state
                elif action == '.':
                    if self.current_state[x, y] == 't':
                        self.current_state[x, y] = '.'
                elif self.current_state[x, y] == 'T':
                    return state
        self.current_state = updated_state
        return state

    def goal_test(self, state):
        for x in range(4):
            for y in range(4):
                #if they are the same, do nothing
                if self.current_state[x, y] == self.goal_sate[x, y]:
                    return state
                else: #if it is not the same, try again
                    self.current_state[x, y] = '.'
            return state

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        raise NotImplementedError


#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
tents_puzzle = Tents(grid_initial, solved_grid, 0, 0)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    tents_puzzle,
    ohio_puzzle,
    franklin_puzzle,
    romania_puzzle,
 #   switch_puzzle,
]
