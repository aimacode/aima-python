import search
from math import(cos, pi)

sumner_map = search.UndirectedGraph(dict(
    # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    # Cottontown=dict(Portland=18),
    # Fairfield=dict(Mitchellville=21, Portland=17),
    # Mitchellville=dict(Portland=7, Fairfield=21),
    Dallas=dict(Austin=50, Houston=100, SanAntonio = 30, Galveston = 10, ElPaso = 200),
    Austin=dict(Dallas=50, Houston=20, ElPaso = 20, SanAntonio = 300, Galveston = 500),
    Houston=dict(Dallas=100, Austin=20, ElPaso = 50, SanAntonio = 200, Galveston = 100),
    ElPaso=dict(Dallas=200, Austin = 20, Houston = 50, SanAntonio = 50, Galveston = 200),
    SanAntonio=dict(Dallas = 30, ElPaso=50, Houston = 200, Austin = 300, Galveston = 500),
    Galveston=dict(Dallas = 10, SanAntonio = 500, ElPaso = 200, Houston = 100, Austin = 300),
))

sumner_puzzle = search.GraphProblem('Dallas', 'SanAntonio', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    sumner_puzzle,
]

