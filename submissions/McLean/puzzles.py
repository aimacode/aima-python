import search
from math import(cos, pi)

#sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
#))

Djibouti_map = search.UndirectedGraph(dict(
    Djibouti=dict(Tadjoura=130, Dikhil=100, AliSabieh=70),
    Tadjoura=dict(Obock=45, Yoboki=90, Djibouti=130, Randa=30),
    AliSabieh=dict(Dikhil=60, Djibouti=70),
    Dikhil=dict(Yoboki=55, Djibouti=100),
    Yoboki=dict(Tadjoura=90,Dikhil=55),
    Randa=dict(Adailou=20, Tadjoura=30),
    Adailou=dict(Guirrori=15, Randa=30),
    Guirrori=dict(Ouaddi=25, Adailou=15),
    Ouaddi=dict(Obock=40, Guirrori=25),
    Obock=dict(Tadjoura=45, Ouaddi=40)
))

#Breadth firs search has a cheaper cost
Djibouti_puzzle = search.GraphProblem('Djibouti', 'Yoboki', Djibouti_map)
#Depth first search has a cheaper cost
Djibouti_puzzle = search.GraphProblem('Djibouti', 'Guirrori', Djibouti_map)

Djibouti_puzzle.description = '''
A map of cities and towns in Djibouti. The link costs are in killometers
'''

myPuzzles = [
Djibouti_puzzle, Djibouti_puzzle
]
