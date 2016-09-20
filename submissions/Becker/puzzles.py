import search
from math import(cos, pi)

# A sample map problem
il_map = search.UndirectedGraph(dict(
    # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    # Cottontown=dict(Portland=18),
    # Fairfield=dict(Mitchellville=21, Portland=17),
    # Mitchellville=dict(Portland=7, Fairfield=21),
    Chicago=dict(Springfield=204, Peoria=169),
    Jacksonville=dict(Springfield=38, Peoria=97, Quincy=79),
    Springfield=dict(Chicago=204, Jacksonville=38, Peoria=68),
    Peoria=dict(Chicago=169, Jacksonville=97, Springfield=68, Quincy=146),
    Quincy=dict(Jacksonville=79, Peoria=146),
))

illinois_puzzle = search.GraphProblem('Chicago', 'Jacksonville', il_map)

illinois_puzzle.label = 'Illinois Map'
illinois_puzzle.description = '''
An abbreviated map of Illinois.
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
    illinois_puzzle,
    switch_puzzle,
]