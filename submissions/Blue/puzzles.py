import search
from math import(cos, pi)

triangle_map = search.UndirectedGraph(dict(
    # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    # Cottontown=dict(Portland=18),
    # Fairfield=dict(Mitchellville=21, Portland=17),
    # Mitchellville=dict(Portland=7, Fairfield=21),
      Apex=dict(Cary=15, ChapelHill=33, FuquayVarina=21, Raleigh=21),
      Cary=dict(Apex=15, ChapelHill=31, Durham=23,Raleigh=31),
      ChapelHill=dict(Apex=33, Cary=31, Durham=24),
      Clayton=dict(Raleigh=26, WakeForest=43),
      Durham=dict(Cary=23, ChapelHill=24, Raleigh=17, WakeForest=37),
      FuquayVarina=dict(Apex=21, Raleigh=32),
      Raleigh=dict(Apex=21, Cary=17, Clayton=26, Durham=31, FuquayVarina=32, WakeForest=29),
      WakeForest=dict(Clayton=43, Durham=37, Raleigh=29),
))

triangle_puzzle_RD = search.GraphProblem('Raleigh', 'Durham',  triangle_map)
# triangle_puzzle_RtoC = search.GraphProblem('Raleigh', 'Cary', triangle_map)
triangle_puzzle_RC = search.GraphProblem('Raleigh', 'ChapelHill', triangle_map) # BFS is better than DFS
triangle_puzzle_CC = search.GraphProblem('ChapelHill', 'Clayton', triangle_map) # UCS is better than BFS

triangle_puzzle_RD.description = '''
An abbreviated map of the Triangle Area of NC.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    # triangle_puzzle,
    triangle_puzzle_RD,
    triangle_puzzle_RC,
    triangle_puzzle_CC,
    # triangle_puzzle_RtoC
]