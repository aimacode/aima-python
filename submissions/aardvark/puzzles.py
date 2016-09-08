import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    Cottontown=dict(Portland=18),
    Fairfield=dict(Mitchellville=21, Portland=17),
    Mitchellville=dict(Portland=7, Fairfield=21),
))

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]