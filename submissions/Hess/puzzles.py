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
    SantaRosa=dict(SanRafael=27, Napa=40),

))

bay_puzzle = search.GraphProblem('PaloAlto', 'Antioch', bay_map)

bay_puzzle.description = '''
An abbreviated map of the Bay Area in California.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    bay_puzzle
]