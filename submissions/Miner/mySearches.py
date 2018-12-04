import search
import numpy as np

initial_grid = []

for i in range(4):
    initial_grid.append([])

for i in range(len(initial_grid)):
    for x in range(4):
        if i == 0 and x == 3:
            initial_grid[i].append('Tree')
        elif i == 1 and x == 1:
            initial_grid[i].append('Tree')
        elif i == 2:
            initial_grid[i].append('Empty')
        elif i == 3 and (x == 0 or x == 3):
            initial_grid[i].append('Tree')
        else:
            initial_grid[i].append('Empty')

goal_grid = []

for i in range(4):
    goal_grid.append([])

for i in range(len(goal_grid)):
    for x in range(4):
        if i == 0:
            if x == 3:
                goal_grid[i].append('Tree')
            if x == 0 or x == 2:
                goal_grid[i].append('Tent')
            elif x == 1:
                goal_grid[i].append('.')
        elif i == 1:
            if x == 1:
                goal_grid[i].append('Tree')
            elif x == 3:
                goal_grid[i].append('Tent')
            elif x == 0 or x == 2:
                goal_grid[i].append('.')
        elif i == 2:
            if x == 0:
                goal_grid[i].append('Tent')
            elif x > 0:
                goal_grid[i].append('.')
        elif i == 3:
            if x == 0 or x == 3:
                goal_grid[i].append('Tree')
            elif x == 2:
                goal_grid[i].append('Tent')
            elif x == 1:
                goal_grid[i].append('.')


def convertToString(grid, grid_dimensions):
    stringified = ''
    for x in range(grid_dimensions):
        for y in range(grid_dimensions):
            stringified += '{}_'.format(grid[x][y])
    return stringified


def stringToList(grid):
    gridSplit = grid.split('_')
    gridSplit.pop(len(gridSplit) - 1)
    every4 = [gridSplit[i: i + 4] for i in range(0, len(gridSplit), 4)]
    return every4


# A sample map problem
auckland_map = search.UndirectedGraph(dict(
    Auckland=dict(MountAlbert=18, MissionBay=22),
    MountAlbert=dict(Auckland=18, Hillsborough=20, Avondale=7),
    MangereEast=dict(Hillsborough=20, Packuranga=20),
    MissionBay=dict(Auckland=22, Hillsborough=31),
    Hillsborough=dict(MissionBay=31, MangereEast=20, MountAlbert=20, Avondale=13, GlenEden=20),
    Avondale=dict(Hillsborough=13, GlenEden=11, MountAlbert=7),
    GlenEden=dict(Hillsborough=20, Avondale=11),
    Epsom=dict(MountAlbert=15, MissionBay=18, Packuranga=17),
    Packuranga=dict(MissionBay=23, Hillsborough=19, MangereEast=20, Otahuhu=15),
    Otahuhu=dict(Hillsborough=16, Pakuranga=15)
))

auckland_puzzle = search.GraphProblem('Auckland', 'GlenEden', auckland_map)

auckland_puzzle.label = 'Auckland, NZ'
auckland_puzzle.description = '''
# An abbreviated map of Auckland, NZ.x
# This map is unique, to the best of my knowledge.
# '''

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


class Tents(search.Problem):
    def __init__(self, initial_state, goal_state, startingX, startingY):
        self.initial = initial_state
        self.goal = goal_state
        self.x_coord = startingX
        self.y_coord = startingY

    def actions(self, state):
        currX = self.x_coord
        currY = self.y_coord

        listState = stringToList(state)
        if listState[currY][currX] == 'Empty':
                # (listState[currY][currX + 1] == 'Tree' or listState[currY + 1][currX] == 'Tree') and \
                # (listState[currY + 1][currX + 1] != 'Tent'):
            return ['Tent']
        elif listState[currY][currX] == 'Tent' or listState[currY][currX] == 'Tree':
            if currX == 3 and currY < 3:
                currX = 0
                currY += 1
                self.x_coord = currX
                self.y_coord = currY
            elif currX < 4:
                self.x_coord = currX + 1
            return ['Tent']

    def result(self, state, action):
        currX = self.x_coord
        currY = self.y_coord

        self.x_coord = currX
        self.y_coord = currY

        newState = stringToList(state)
        goalState = stringToList(self.goal)

        if action == 'Tent':
            if currX == 3 and currY < 3:
                self.x_coord = 0
                self.y_coord = currY + 1
            elif currX < 4:
                self.x_coord = currX + 1
            newState[currY][currX] = 'Tent'
            if newState[currY][currX] == goalState[currY][currX]:
                return convertToString(newState, 4)
            else:
                newState[currY][currX] = '.'
                return convertToString(newState, 4)
        #     # return convertToString(newState, 4)
        # elif action == 'Next':
        #     return state

    def goal_test(self, state):
        # print('goal test', state == self.goal)
        return state == self.goal


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



# #swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
# switch_puzzle = LightSwitch('off')
# switch_puzzle.label = 'Light Switch'


tents_puzzle = Tents(convertToString(initial_grid, 4), convertToString(goal_grid, 4), 0, 0)
tents_puzzle.label = 'Tents'

mySearches = [
 #   swiss_puzzle,
    romania_puzzle,
    tents_puzzle,
    # switch_puzzle,
    auckland_puzzle
]

mySearchMethods = []
