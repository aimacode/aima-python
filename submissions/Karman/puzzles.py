import search
from math import(cos, pi)


tuscany_map = search.UndirectedGraph(dict(
Florence=dict(Prato=24, Montevarchi=51, Sienna=75, Empoli=31),
Prato=dict(Pistoia=26, Florence=24, Empoli = 29.3),
Pistoia=dict(Prato=26, Lucca=43),
Lucca=dict(Pistoia=43, Pisa=18, Pontedera = 29, SanMiniato = 44.3),
Pisa=dict(Lucca=18, Pontedera=33, Cecina=63.3),
Pontedera=dict(Pisa=33, SanMiniato=22, Lucca = 29, Volterra=45.5),
SanMiniato=dict(Pontedera=22, Empoli=11, Lucca = 44.3),
Empoli=dict(SanMiniato=11, Florence=31, Sienna = 76.6, Prato = 29.3),
Montevarchi=dict(Florence=51, Arezzo=37, Sienna = 54.4),
Arezzo=dict(Montevarchi=37, Sienna=74.6),
Sienna=dict(Florence=75, Arezzo =74.6, Empoli =76.6, Montevarchi = 54.4, Volterra = 54),
Volterra=dict(Sienna=54, Pontedera=45.5, Cecina=39),
Cecina=dict(Volterra=39, Pisa=63.3)

))

tuscany_map.locations = dict(
    Florence=(43.77, 11.25), Prato=(43.87,11.10),
    Pistoia=(43.93, 10.90), Lucca=(43.83, 10.49),
    Pisa=(43.72, 10.39), Pontedera=(43.66, 10.62),
    SanMiniato=(43.68, 10.85), Empoli=(43.72, 10.95),
    Montevarchi=(43.53, 11.56), Arezzo=(43.46, 11.89),
    Sienna=(43.32, 11.33), Volterra=(43.40, 10.86),
    Cecina=(43.31,10.51),
)
tuscany_puzzle = search.GraphProblem('Sienna', 'Lucca', tuscany_map)
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