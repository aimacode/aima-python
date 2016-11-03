import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
   # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   # Cottontown=dict(Portland=18),
   # Fairfield=dict(Mitchellville=21, Portland=17),
   # Mitchellville=dict(Portland=7, Fairfield=21),
#))

#sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

#sumner_puzzle.label = 'Sumner Map'
#sumner_puzzle.description = '''
#An abbreviated map of Sumner County, TN.
#This map is unique, to the best of my knowledge.

# My Northern Il and Indiana map
northernIl_map = search.UndirectedGraph(dict(
    Rockford=dict(CrystalLake=42),
    CrystalLake=dict(Chicago=53, Aurora=35),
    Chicago=dict(Aurora=41, Hammond=27, Detriot=284),
    Detriot=dict(Indianapolis=287),
    Aurora=dict(Chicago=41, Kankakee=81),
    Hammond=dict(SouthBend=71),
    Kankakee=dict(Champaign=160),
    Champaign=dict(Lafayette=91),
    Lafayette=dict(Indianapolis=62),
    Indianapolis=dict(SouthBend=150),
))
northernIl_puzzle2 = search.GraphProblem('Rockford', 'Indianapolis', northernIl_map)
northernIl_map.locations = dict(
    Rockford = (42, 89),
    CrystalLake= (42, 88),
    Chicago= (41, 87),
    Detriot= (42, 83),
    Aurora= (41, 88),
    Hammond= (41, 87),
    Champaign= (40, 88),
    Lafayette = (40, 86),
    Indianapolis= (39, 86),
)


northernIl_puzzle = search.GraphProblem('Rockford', 'SouthBend', northernIl_map)
#northernIl_puzzle shows breadth_first_search is lower than depth_first_search

northernIl_map.label = 'Northern Midwest Map'
northernIl_map.description= 'It is a map of the Northern Midwest'
#An abbreviated map of Northern Midwest

''

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
    northernIl_puzzle,
    switch_puzzle,
    northernIl_puzzle2,
]