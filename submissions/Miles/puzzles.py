import search
from math import(cos, pi)

# sumner_map = search.UndirectedGraph(dict(
   # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   # Cottontown=dict(Portland=18),
   # Fairfield=dict(Mitchellville=21, Portland=17),
   # Mitchellville=dict(Portland=7, Fairfield=21),
# ))
alabama_map = search.UndirectedGraph(dict(
    Birmingham=dict(Tuscaloosa=45, Auburn=120, Montgomery=86, Huntsville=90, Mobile=219, Dothan=197),
    Tuscaloosa=dict(Birmingham=45, Auburn=160, Montgomery=110, Huntsville=140, Mobile=211, Dothan=227),
    Auburn=dict(Birmingham=120, Tuscaloosa=160, Montgomery=57, Huntsville=212, Mobile=195, Dothan=130),
    Huntsville=dict(Birmingham=90, Tuscaloosa=140, Montgomery=166, Auburn=212, Mobile=302, Dothan=279),
    Montgomery=dict(Birmingham=86, Tuscaloosa=110, Auburn=57, Huntsville=166, Mobile=144, Dothan=120),
    Mobile=dict(Birmingham=219, Tuscaloosa=211, Auburn=195, Montgomery=144, Huntsville=302, Dothan=184),
    Dothan=dict(Birmingham=197, Tuscaloosa=227, Auburn=130, Montgomery=120, Huntsville=279, Mobile=184),
    Gardendale=dict(Birmingham=21),
    Fairhope=dict(Mobile=26, Birmingham=237)



))
# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
alabama_puzzle = search.GraphProblem('Fairhope', 'Auburn', alabama_map)
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''

alabama_puzzle.description = '''
An abbreviated map of Middle Alabama.
This map is unique, to the best of my knowledge.
'''
# myPuzzles = [
    # sumner_puzzle,
# ]

myPuzzles = [
    alabama_puzzle,
]