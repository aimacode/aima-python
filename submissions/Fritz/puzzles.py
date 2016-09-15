import search
from math import(cos, pi)

AK_map = search.UndirectedGraph(dict(
     #Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
     #Cottontown=dict(Portland=18),
     #Fairfield=dict(Mitchellville=21, Portland=17),
     #Mitchellville=dict(Portland=7, Fairfield=21),
    Angoon=dict(Juneau=10, Hoonah=9, Wrangell=18, PortProtection=15),
    ElfinCove=dict(Juneau=11, Gustavus=4),
    Gustavus=dict(Juneau=8, ElfinCove=4, MudBay=9),
    Haines=dict(Skagway=3, Juneau=12, MudBay=2),
    Hoonah=dict(Juneau=8, Angoon=9),
    Juneau=dict(Haines=12, Skagway=14, Hoonah=8, ElfinCove=11, Gustavus= 8, Angoon=10),
    MudBay=dict(Gustavus=9, Haines=2),
    PortProtection=dict(Wrangell=9, Angoon=15),
    Skagway=dict(Juneau=14, Haines=3),
    Wrangell=dict(Angoon=18, PortProtection=9),
))

AK_puzzle = search.GraphProblem('Haines', 'ElfinCove', AK_map) #This instance yields a better solution with UCS than BFS
AK_puzzle = search.GraphProblem('MudBay', 'Hoonah', AK_map) #This instance yields a better solution with BFS than DFS

AK_puzzle.description = '''
An abbreviated map of south east Alaska.
This map is unique, to the best of my knowledge.
'''

myPuzzles = [
    AK_puzzle,
]