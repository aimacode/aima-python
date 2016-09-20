import search
from math import(cos, pi)


tuscany_map = search.UndirectedGraph(dict(
Florence=dict(Prato=24, Montevarchi=51, Sienna=75, Empoli=31),
Prato=dict(Pistoia=26, Florence=24),
Pistoia=dict(Prato=26, Lucca=43),
Lucca=dict(Pistoia=43, Pisa=18),
Pisa=dict(Lucca=18, Pontedera=33),
Pontedera=dict(Pisa=33, SanMiniato=22),
SanMiniato=dict(Pontedera=22, Empoli=11),
Empoli=dict(SanMiniato=11, Florence=31),
Montevarchi=dict(Florence=51, Arezzo=37),
Arezzo=dict(Montevarchi=37, Sienna=75),
Sienna=dict(Florence=75, Arezzo = 74.6),
))

tuscany_puzzle = search.GraphProblem('Florence', 'Pisa', tuscany_map)
tuscany_puzzle.label = 'Tuscany Map'
tuscany_puzzle.description = '''
an abbreviated map of the Tuscany region of Italy
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
    tuscany_puzzle,
    switch_puzzle,
]