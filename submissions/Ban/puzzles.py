import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    Dublin=dict(Mullingar=67),
    Mullingar=dict(Naas=59, Dublin=67),
    Naas=dict(Mullingar=59),
    Kells=dict(Dublin=61, Mullingar=44),
    Arklow=dict(Naas=80, Dublin=89),
))

sumner_puzzle = search.GraphProblem('Mullingar', 'Kells', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Eastern Ireland.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]