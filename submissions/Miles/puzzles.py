import search
from math import (cos, pi)

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
alabama_map.locations = dict(
    Birmingham=(50, 300), Tuscaloosa=(20, 270), Auburn=(50, 180),
    Montgomery=(45, 214), Huntsville=(50, 390), Mobile=(10, 85),
    Dothan=(100, 170), Gardendale=(50, 321), Fairhope=(10, 59))
alabama_puzzle = search.GraphProblem('Fairhope', 'Tuscaloosa', alabama_map)
alabama_puzzle.description = '''
An abbreviated map of Middle Alabama.
This map is unique, to the best of my knowledge.
'''


# A trivial Problem definition of connect four
# The goal is to get either 4 x's in a row or 4 o's in a row
# The x's and o's represent the colors red and yellow

class ConnectFour(search.Problem):
    def actions(self, state):

                    # return connect_four

            Red = 'X'  # the player
            Yellow = 'O'  # the computer
            player1 = 'Winner'
            state = ConnectFour([['O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O'],
                                 ])
            return state

    def result(self, state, action):
        if action == 'X':
            self.rows
            return 'X'
        else:
            self.columns
            return 'O'


    def goal_test(self, state):
        GOAL = ('X', 'X', 'X', 'X')
        GOAL2 = ('O', 'O', 'O', 'O')
        return state == GOAL or GOAL2


    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


miles_puzzle = ConnectFour('X')
miles_puzzle.label = 'Connect Four'

myPuzzles = [
    alabama_puzzle,
    miles_puzzle

]
