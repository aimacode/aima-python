import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
    Amsterdam=dict(Utrecht=53,Hague=60, Zwolle=112, Haarlem=20),
    Arnhem=dict(Eindhoven=84, Maastricht=167),
    Breda=dict(Rotterdam=50, Heerlen=148),
    Eindhoven=dict(Rotterdam=110, Utrecht=92, Arnhem=84, Maastricht=88),
    Haarlem=dict(Amsterdam=20, Hague=52),
    Hague=dict(Ultrecht=68, Rotterdam=30, Amsterdam=60, Haarlem=52),
    Heerlen=dict(Maastricht=25, Breda=148),
    Maastricht=dict(Eindhoven=88, Arnhem=167, Heerlen=25),
    Rotterdam=dict(Utrecht=61, Hague=30, Eindhoven=110, Breda=50),
    Utrecht=dict(Amsterdam=53,Zwolle=90, Hague=68, Eindhoven=92, Rotterdam=61),
    Zwolle=dict(Utrecht=90, Amsterdam=112),
))

sumner_puzzle = search.GraphProblem('Hague', 'Maastricht', sumner_map)

sumner_puzzle.label = 'Netherlands Map'
sumner_puzzle.description = '''
An abbreviated map of the Netherlands.
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