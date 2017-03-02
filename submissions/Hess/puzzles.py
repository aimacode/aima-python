import search
from math import(cos, pi)

bay_map = search.UndirectedGraph(dict(

    Antioch=dict(Berkeley=35, Richmond=17),
    Berkeley=dict(Antioch=35, Oakland=5),
    Hayward=dict(SanJose=34, Oakland=17),
    Napa=dict(Richmond=32, SantaRosa=40),
    Oakland=dict(SanFrancisco=12, Berkeley=5, Hayward=17, Richmond=12),
    PaloAlto=dict(SanFrancisco=33, SanJose=10),
    Richmond=dict(Oakland=12, Antioch=17, SanRafael=13, Napa=32),
    SanFrancisco=dict(SanRafael=18, Oakland=12, PaloAlto=33),
    SanJose=dict(PaloAlto=10),
    SanRafael=dict(SantaRosa=27, SanFrancisco=18, Richmond=13),
    SantaRosa=dict(SanRafael=27, Napa=40)))

bay_map.locations = dict(
    Antioch=(38.0049214, -121.805789),
    Berkeley=(37.8715926, -122.272747),
    Hayward=(37.6688205, -122.0807964),
    Napa=(38.2975381, -122.286865),
    Oakland=(37.8043637, -122.2711137),
    PaloAlto=(37.4418834, -122.1430195),
    Richmond=(37.9357576, -122.3477486),
    SanFrancisco=(37.7749295, -122.4194155),
    SanJose=(37.3382082, -121.8863286),
    SanRafael=(37.9735346, -122.5310874),
    SantaRosa=(38.440429, -122.7140548))


bay_puzzle = search.GraphProblem('PaloAlto', 'Antioch', bay_map)
bay_puzzle2 = search.GraphProblem('Napa', 'PaloAlto', bay_map)

bay_puzzle.description = '''
An abbreviated map of the Bay Area in California.
This map is unique, to the best of my knowledge.
'''


tentsPuzzle = ([['^', 'T', '0', '0'],
                ['0', '0', 'T', '0'],
                ['0', '0', '0', 'T'],
                ['T', '', '0', '0'],
                ])

class Tents(search.Problem):
    def actions(self, state):
        tree = 'T',
        tent = '^',
        grass = '0',

        return state

    def result(self, state, action):
        if action == '^':
            self.rows
            return '^'
        else:
            self.columns
            return 'off'

    def goal_test(self, state):
        GOAL = ('^', 'T', '0', '0') or ('0', 'T', '^', '0')
        GOAL2 = ('0', '^', 'T', '0') or ('0', '0', 'T', '^')
        GOAL3 = ('0', '0', '^', 'T')
        GOAL4 = ('T', '^', '0', '0')
        return state == GOAL and GOAL2 and GOAL3 and GOAL4

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

tents_puzzle = Tents('^')
tents_puzzle.label = 'Tents'

myPuzzles = [
    bay_puzzle,
    bay_puzzle2,
    tents_puzzle,
]