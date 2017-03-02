import search
from math import(cos, pi)

# A sample map problem
sweden_map = search.UndirectedGraph(dict(
    Stockholm=dict(Uppsala=71, Umeå=637, Malmö=271),
    Gothenburg=dict(Örebro=218, Östersund=779, Stockholm=471, Lund=264),
    Malmö=dict(Gothenburg=271, Örebro=502, Helsingborg=64, Lund=19),
    Uppsala=dict(Umeå=571, Stockholm=72),
    Örebro=dict(Stockholm=202, Linköping=122, Umeå=701),
    Linköping=dict(Stockholm=200, Uppsala=265, Norrköping=42, Lund=409),
    Helsingborg=dict(Gothenburg=215, Stockholm=556, Linköping=360),
    Jönköping=dict(Örebro=213, Stockholm=324, Norrköping=167),
    Norrköping=dict(Stockholm=163, Umeå=795, Linköping=42),
    Lund=dict(Stockholm=601, Malmö=19, Umeå=1236),
    Umeå=dict(Östersund=363, Stockholm=639),
    Östersund=dict(Umeå=363, Gothenburg=779, Lund=1156),
))

sweden_puzzle = search.GraphProblem('Malmö', 'Umeå', sweden_map)

sweden_puzzle.label = 'Map of Sweden'
sweden_puzzle.description = '''
An abbreviated map of Sweden.
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
    sweden_puzzle,
    switch_puzzle,
]