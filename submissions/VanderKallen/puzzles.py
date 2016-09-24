import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
    Amsterdam=dict(Utrecht=53,Hague=60, Zwolle=112, Haarlem=20),
    Arnhem=dict(Eindhoven=84, Maastricht=167),
    Breda=dict(Rotterdam=50, Heerlen=148),
    Eindhoven=dict(Rotterdam=110, Utrecht=92, Arnhem=84, Maastricht=88),
    Haarlem=dict(Amsterdam=20, Hague=52),
    Hague=dict(Utrecht=68, Rotterdam=30, Amsterdam=60, Haarlem=52),
    Heerlen=dict(Maastricht=25, Breda=148),
    Maastricht=dict(Eindhoven=88, Arnhem=167, Heerlen=25),
    Rotterdam=dict(Utrecht=61, Hague=30, Eindhoven=110, Breda=50),
    Utrecht=dict(Amsterdam=53, Zwolle=90, Hague=68, Eindhoven=92, Rotterdam=61),
    Zwolle=dict(Utrecht=90, Amsterdam=112),
))

sumner_map.locations = dict(
    Amsterdam=(217, 386), Arnhem=(378, 314), Breda=(209, 201),
    Eindhoven=(322, 169), Haarlem=(201, 389), Hague=(137, 330),
    Heerlen=(286, 40), Maastricht=(346, 32), Rotterdam=(177,298), Utrecht=(239, 333), Zwolle=(394, 422))

sumner_puzzle = search.GraphProblem('Hague', 'Maastricht', sumner_map)
sumner_puzzle2 = search.GraphProblem('Heerlen', 'Zwolle', sumner_map)


sumner_puzzle.label = 'BestFS > DFS, and UCS = A*, less moves'
sumner_puzzle2.label = 'BFS > BestFS'
sumner_puzzle.description = '''
An abbreviated map of the Netherlands.
This map is unique, to the best of my knowledge.
'''

# A trivial Problem definition
class Twiddle(search.Problem):
    def actions(self, state):
        return [(0,0,'cc'),(0,0,'cw'),(0,2,'cc'), (0,2,'cw')]

    def result(self, state, action):
        newState = state.copy()
        if action == (0,0,'cc'):
            newState[0][0]= state[0][1]
            newState[0][1]= state[1][1]
            newState[1][0]= state[0][0]
            newState[1][1]= state[1][0]
            return newState
        else:
            if action == (0,0,'cw'):
                newState[0][0] = state[1][0]
                newState[0][1] = state[0][0]
                newState[1][0] = state[1][1]
                newState[1][1] = state[0][1]
                return newState
            else:
                if action == (0,2,'cc'):
                    newState[0][1] = state[0][2]
                    newState[0][2] = state[1][2]
                    newState[1][1] = state[0][1]
                    newState[1][2] = state[1][1]
                    return newState
                else:
                    if action == (0,2,'cw'):
                        newState[0][1] = state[1][1]
                        newState[0][2] = state[0][1]
                        newState[1][1] = state[1][2]
                        newState[1][2] = state[0][2]
                        return newState

    def goal_test(self, state):
        return state == [['1','2','3'],['4','5','6']]

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

twiddle_puzzle = Twiddle([['1','2','3'],['4','5','6']])
twiddle_puzzle.label = 'Twiddle'

myPuzzles = [
    sumner_puzzle,
    sumner_puzzle2,
    twiddle_puzzle,
]