import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Kirkwood=dict(Webster=10, Clayton=17, St_Louis=21),
    St_Louis=dict(Clayton=12),
    Clayton=dict(Webster=14, Kirkwood=17),
    Webster=dict(Kirkwood=10, Clayton=14),
))

sumner_puzzle = search.GraphProblem('Webster', 'St_Louis', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]