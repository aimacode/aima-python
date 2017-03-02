import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Dublin=dict(Mullingar=79),
    Mullingar=dict(Naas=70, Dublin=79),
    Naas=dict(Mullingar=70, Carlow=58),
    Kells=dict(Dublin=65, Mullingar=42),
    Arklow=dict(Naas=97, Dublin=71),
    Carlow=dict(Naas=58),
))

sumner_map.locations = dict(
    Dublin=(69, 42),
    Mullingar=(0, 22),
    Naas=(41, 53),
    Kells=(30, 0),
    Arklow=(75, 101),
    Carlow=(62, 25)
)

sumner_puzzle = search.GraphProblem('Mullingar', 'Carlow', sumner_map)
sumner_puzzle.label = 'Sumner Map'
sumner_puzzle.description = '''
An abbreviated map of Eastern Ireland.
This map is unique, to the best of my knowledge.
'''
sumner_puzzle2 = search.GraphProblem('Mullingar', 'Arklow', sumner_map)
sumner_puzzle2.label = 'Sumner Map 2'
sumner_puzzle2.description = '''
An abbreviated map of Eastern Ireland.
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

switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'




# This is basically sodoku, so... It is just sodoku.

puzzle = ([ [0,1,0,4],
            [4,0,1,0],
            [3,0,2,0],
            [0,2,0,3]])

class Solo(search.Problem):
    def actions(self, state):
        x = 0
        y = 0
        for row in puzzle:
            y += 1
            for cell in row:
                x += 1
                if cell == 0:
                    return [[x,y,1],[x,y,2],[x,y,3],[x,y,4]]

    def result(self, state, action):
        state[action[1]][action[0]] = action[2]
        return state

    def goal_test(self, state):
        blanks = 0
        for row in puzzle:
            for cell in row:
                if cell == 0:
                    blanks += 1
        if blanks == 0:
            return state == 'complete'
        else:
            return state == 'incomplete'

    def valid_input(self, state):
        # this is the validity test, which will test for valid inputs and say whether or not a result is valid
        return state

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


solo_puzzle = Solo(puzzle)
solo_puzzle.label = 'Solo Game'

myPuzzles = [
    sumner_puzzle,
    sumner_puzzle2,
    switch_puzzle,
    # solo_puzzle,
]