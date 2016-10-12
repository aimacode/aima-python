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

sumner_puzzle.label = 'Sumner Map'
sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
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

myPuzzles = [
    sumner_puzzle,
    switch_puzzle,
]