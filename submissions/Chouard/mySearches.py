import search
from math import (cos, pi)

# A sample map problem

# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))
#
# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
#
# sumner_puzzle.label = 'Sumner'
#
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''
# romania_map = search.UndirectedGraph(dict(
#     A=dict(Z=75,S=140,T=118),
#     Z=dict(O=71,A=75),
#     S=dict(O=151,R=80,F=99),
#     T=dict(A=118,L=111),
#     O=dict(Z=71,S=151),
#     L=dict(T=111,M=70),
#     M=dict(L=70,D=75),
#     D=dict(M=75,C=120),
#     R=dict(S=80,C=146,P=97),
#     C=dict(R=146,P=138,D=120),
#     F=dict(S=99,B=211),
#     P=dict(R=97,C=138,B=101),
#     B=dict(G=90,P=101,F=211),
# ))
#
# romania_puzzle = search.GraphProblem('A', 'B', romania_map)
#
# romania_puzzle.label = 'Romania'
# romania_puzzle.description = '''
# The simplified map of Romania, per
# Russall & Norvig, 3rd Ed., p. 68.
# '''

cotes_dazure_map = search.UndirectedGraph(dict(
    Marseille=dict(Manosque=65, Aubagne=25),
    Aubagne=dict(Marseille=25, Toulon=34, Brignoles=46),
    Toulon=dict(SaintTropez=71, Brignoles=41, Aubagne=34),
    Brignoles=dict(Aubagne=34, Toulon=41, Frejus=46, Draguignan=56),
    Manosque=dict(Marseille=65, Gap=74),
    SaintTropez=dict(Toulon=71, Frejus=49),
    Frejus=dict(SaintTropez=49, Brignoles=46, Nice=53),
    Draguignan=dict(Brignoles=56, Digne=113),
    Gap=dict(Manosque=74, Digne=93),
    Digne=dict(Draguignan=113, Gap=93, Nice=146),
    Nice=dict(Frejus=53, Digne=146)
))

cotes_dazure_puzzle = search.GraphProblem('Marseille', 'Nice', cotes_dazure_map)

cotes_dazure_puzzle.label = "Cotes d'Azur"

cotes_dazure_puzzle.description = '''
An abbreviated map of Provence-Alpes-Côte d’Azur, France.
This map is unique, to the best of my knowledge.
'''

class OneTwoThree(search.Problem):
    def __init__(self, initial, goal):
        initial_string = self.state_to_string(initial)
        search.Problem.__init__(self, initial_string, goal)
        self.size = len(initial)

    def actions(self, state):
        return [1, 2, 3]

    def result(self, state, action):
        state = self.string_to_state(state)
        position = (0, 0)
        while state[position[0]][position[1]] != 0:
            if position[0] == self.size - 1 and position[1] == self.size - 1:
                return self.state_to_string(state)
            position = self.inc_position(position[0], position[1])
        nextState = list(list(li) for li in state)
        nextState[position[0]][position[1]] = action
        return self.state_to_string(nextState)

    def inc_position(self, row, col):
        if col == self.size - 1:
            col = 0
            row = row + 1
        else:
            col += 1
        return row, col

    def goal_test(self, state):
        state = self.string_to_state(state)
        for m in range(0, self.size):
            for n in range(0, self.size):
                if state[m][n] == self.goal[m][n]:
                    continue
                return False
        return True

    def h(self, node):
        state = self.string_to_state(node.state)
        score = 0
        for m in range(0, self.size):
            for n in range(0, self.size):
                score += self.count_similar_around(state, m, n)
        # divide the sum of the scores for the squares by the total number of squares for a number less than 1.
        total_score = score / self.size ** 2
        return 1 - total_score

    def count_similar_around(self, state, row, col):
        count = 0
        if state[row][col] == 0:
            return 0
        # check right
        if not (col + 1 == self.size):
            if state[row][col + 1] == state[row][col]:
                count += 1
        # check left
        if not (col - 1 < 0):
            if state[row][col - 1] == state[row][col]:
                count += 1
        # check top
        if not (row - 1 < 0):
            if state[row - 1][col] == state[row][col]:
                count += 1
        # check bottom
        if not (row + 1 == self.size):
            if state[row + 1][col] == state[row][col]:
                count += 1
        # score based on the number of identical numbers touching the square.
        # 2 should have 1 two next to it, etc.
        score = (count + 1) / state[row][col]
        if score > 1:
            return 0
        return score

    def state_to_string(self, state):
        rows = []
        for row in state:
            row = map(lambda x: str(x), row)
            rows.append(''.join(row))
        return '|'.join(rows)

    def string_to_state(self, state_string):
        rows = state_string.split('|')
        state = []
        for row in rows:
            row_split = list(row)
            row = [int(i) for i in row_split]
            state.append(row)
        return state


ott_initial = [[1, 3, 2], [3, 0, 2], [2, 2, 1]]
ott_goal = [[1, 3, 2], [3, 3, 2], [2, 2, 1]]
ott_puzzle_1 = OneTwoThree(initial=ott_initial, goal=ott_goal)
ott_puzzle_1.label = 'One Two Three 1 Step'

ott_initial = [[1, 3, 0], [3, 0, 2], [2, 2, 1]]
ott_goal = [[1, 3, 2], [3, 3, 2], [2, 2, 1]]
ott_puzzle_2 = OneTwoThree(initial=ott_initial, goal=ott_goal)
ott_puzzle_2.label = 'One Two Three 2 Steps'

ott_initial = [[0, 0, 0], [3, 0, 2], [2, 2, 1]]
ott_goal = [[1, 3, 2], [3, 3, 2], [2, 2, 1]]
ott_puzzle_4 = OneTwoThree(initial=ott_initial, goal=ott_goal)
ott_puzzle_4.label = 'One Two Three 4 Steps'

ott_initial = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
ott_goal = [[1, 3, 2], [3, 3, 2], [2, 2, 1]]
ott_puzzle_b = OneTwoThree(initial=ott_initial, goal=ott_goal)
ott_puzzle_b.label = 'One Two Three Big'

mySearches = [
    #   swiss_puzzle,
    # sumner_puzzle,
    # romania_puzzle,
    cotes_dazure_puzzle,
    ott_puzzle_1,
    ott_puzzle_2,
    ott_puzzle_4,
    ott_puzzle_b,
]

import random


def flounder(problem, giveup=10000):
    'The worst way to solve a problem'
    node = search.Node(problem.initial)
    count = 0
    while not problem.goal_test(node.state):
        count += 1
        if count >= giveup:
            return None
        children = node.expand(problem)
        node = random.choice(children)
    return node


mySearchMethods = [
    # flounder
]
