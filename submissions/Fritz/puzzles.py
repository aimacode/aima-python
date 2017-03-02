import search
from math import(cos, pi)

AK_map = search.UndirectedGraph(dict(
     #Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
     #Cottontown=dict(Portland=18),
     #Fairfield=dict(Mitchellville=21, Portland=17),
     #Mitchellville=dict(Portland=7, Fairfield=21),
    Angoon=dict(Juneau=10, Hoonah=9, Wrangell=18, PortProtection=15),
    ElfinCove=dict(Juneau=11, Gustavus=4),
    Gustavus=dict(Juneau=8, ElfinCove=4, MudBay=9),
    Haines=dict(Skagway=3, Juneau=12, MudBay=2),
    Hoonah=dict(Juneau=8, Angoon=9),
    Juneau=dict(Haines=12, Skagway=14, Hoonah=8, ElfinCove=11, Gustavus= 8, Angoon=10),
    MudBay=dict(Gustavus=9, Haines=2),
    PortProtection=dict(Wrangell=9, Angoon=15),
    Skagway=dict(Juneau=14, Haines=3),
    Wrangell=dict(Angoon=18, PortProtection=9),
))
AK_map.locations = dict(
    angoon=(498, 583), ElfinCove=(191, 346), Gustavus=(408, 738),
            Haines=(231, 451), Hoonah=(108,445), Juneau=(291, 417),
            MudBay=(156, 372), PortPrtection=(323, 586),
            Skagway=(446, 322), Wrangell=(467, 378))

AK_puzzle = search.GraphProblem('Haines', 'ElfinCove', AK_map) #This instance yields a better solution with UCS than BFS
AK_puzzle1 = search.GraphProblem('MudBay', 'Hoonah', AK_map) #This instance yields a better solution with BFS than DFS

AK_puzzle.description = '''
An abbreviated map of south east Alaska.
This map is unique, to the best of my knowledge.
'''

# A trivial Problem definition
class cube(search.Problem):

    def actions(self, state):
        return ['moveright', 'moveleft', 'moveup', 'movedown']

    def result(self, state, action):
        if action == 'moveright':
            action = 'moveup'
            self
            return 'rollright'
        else:
             if action == 'moveleft':
              #   action = 'movedown'
               #  self
                 return 'rollleft'
             else:
                 if action == 'moveup':
                     return 'rollup'
                 else:
                     if action == 'movedown':
                         return 'rolldown'


    def goal_test(self, state):
        return state == 'rolldown'

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

cube_puzzle = cube('rollright')
cube_puzzle.label = 'Cube Puzzle'

myPuzzles = [
    cube_puzzle,
]