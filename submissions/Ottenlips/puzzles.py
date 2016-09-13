import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Kirkwood=dict(Webster=7, Clayton=17, St_Louis=18),
    St_Louis=dict(Clayton=18),
    Clayton=dict(Webster=21, Kirkwood=17),
    Webster=dict(Kirkwood=7, Clayton=21),
))

sumner_puzzle = search.GraphProblem('Webster', 'St_Louis', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]