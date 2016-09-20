import search
from math import(cos, pi)

stl_map = search.UndirectedGraph(dict(
    Kirkwood=dict(Webster=10, Clayton=17, MapleWood=17, Glendale=7),
    St_Louis=dict(Clayton=12),
    Glendale=dict(St_Louis=19),
    MapleWood=dict(St_Louis=11),
    Clayton=dict(Webster=14, St_Louis=12, Kirkwood=17),
    Webster=dict(Kirkwood=10, Clayton=14),
))


stl_puzzle = search.GraphProblem('Kirkwood', 'St_Louis', stl_map)

stl_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    stl_puzzle,
]