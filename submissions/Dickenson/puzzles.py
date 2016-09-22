import search
from math import(cos, pi)

# A sample map problem
zimbabwe_map = search.UndirectedGraph(dict(
    Harare=dict(MountDarwin=118, Kadoma=102, Chikore=138, Mutare=197),
    Kadoma=dict(Harare=102, Kadoma=107),
    Gweru=dict(Kadoma=107, Bulawayo=99),
    Bulawayo=dict(Gweru=99, Lubimbi=183),
    Lubimbi=dict(Gweru=267, Bulawayo=183),
    Mutare=dict(Harare=197, Nyanga=98),
    MountDarwin=dict(Harare=118),
    Nyanga=dict(Chikore=106, Mutare=98),
    Chikore=dict(Harare=138, Nyanga=106),
))

sumner_puzzle = search.GraphProblem('Harare', 'Mutare', zimbabwe_map)

sumner_puzzle.label = 'Zimbabwe Map'
sumner_puzzle.description = '''
An abbreviated map of several cities in Zimbabwe.
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