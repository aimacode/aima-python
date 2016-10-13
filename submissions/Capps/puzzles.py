import search
from math import(cos, pi)

# A sample map problem
Butte_map = search.UndirectedGraph(dict(
    Missoula=dict(Lolo=21, Philipsburg=71, Helena=111, Anaconda=94),
    Lolo=dict(Hamilton=40, Missoula=21),
    Hamilton=dict(Lolo=40, Butte=159, Sula=38),
    Salmon=dict(Cobalt=76, Bannack=128),
    DeerLodge=dict(Helena=64, Dillon=110, Anaconda=29),
    Anaconda=dict(Missoula=94, Sula=116, Dillon=70, DeerLodge=29),
    Butte=dict(Hamilton=159, Bozeman=81, Bannack=81, Astley=110),
    Dillon=dict(DeerLodge=110, Polaris=41, Anaconda=70),
    Bozeman=dict(Checkerboard=108, Helena=97, Butte=81),
    Polaris=dict(Dillon=41, Bannack=24, Jackson=24, Achlin=40),
    Jackson=dict(Sula=56, Polaris=24),
    Sula=dict(Jackson=56, Hamilton=38, Anaconda=116),
    Philipsburg=dict(Missoula=71),
    Cobalt=dict(Salmon=76),
    Bannack=dict(Salmon=128, Polaris=24, Butte=81),
    Checkerboard=dict(Bozeman=108),
    Astley=dict(Bentwood=32, Butte=110),
    Bentwood=dict(Astley=32, Achlin=70),
    Achlin=dict(Polaris=40, Bentwood=70),
))

#Butte_map = search.GraphProblem('Butte', 'DeerLodge', Butte_map) #BFS > DFS
Butte_map = search.GraphProblem('Butte', 'Anaconda', Butte_map) #UCS > BFS > DFS    where ">" means better(faster)

Butte_map.label = 'Butte Montana'
Butte_map.description = '''
A map consisting of multiple counties near Butte, Montana. Reaches into Idaho.
This map is unique, to the best of my knowledge.
'''

# A trivial Problem definition
class Twiddle(search.Problem):
    def actions(self, state):
        return [(0, 0, 'cc'), (0, 0, 'cw'), (0, 2, 'cc'), (0, 2, 'cw')]

    def result(self, state, action):
        newState = state.copy()
        if action == (0, 0, 'cc'):
            newState[0][0] = state[0][1]
            newState[0][1] = state[1][1]
            newState[1][0] = state[0][0]
            newState[1][1] = state[1][0]
            return newState
        else:
            if action == (0, 0, 'cw'):
                newState[0][0] = state[1][0]
                newState[0][1] = state[0][0]
                newState[1][0] = state[1][1]
                newState[1][1] = state[0][1]
                return newState
            else:
                if action == (0, 2, 'cc'):
                    newState[0][1] = state[0][2]
                    newState[0][2] = state[1][2]
                    newState[1][1] = state[0][1]
                    newState[1][2] = state[1][1]
                    return newState
                else:
                    if action == (0, 2, 'cw'):
                        newState[0][1] = state[1][1]
                        newState[0][2] = state[0][1]
                        newState[1][1] = state[1][2]
                        newState[1][2] = state[0][2]
                        return newState

    def goal_test(self, state):
        return state == [['1', '2', '3'], ['4', '5', '6']]

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


twiddle_puzzle = Twiddle([['1', '2', '3'], ['4', '5', '6']])
twiddle_puzzle.label = 'Twiddle Puzzle TRICKS'

myPuzzles = [
    #Butte_map,
    twiddle_puzzle,
]