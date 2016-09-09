import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Dublin=dict(Mullingar=67, Nass=41),
    Mullingar=dict(Dublin=67, Naas=59),
    Naas=dict(Dublin=41, Mullingar=59)
))

sumner_puzzle = search.GraphProblem('Dublin', 'Mullinger', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Eastern Ireland.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]