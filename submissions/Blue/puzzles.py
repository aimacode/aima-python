import search
from math import(cos, pi)

triangle_map = search.UndirectedGraph(dict(
    # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    # Cottontown=dict(Portland=18),
    # Fairfield=dict(Mitchellville=21, Portland=17),
    # Mitchellville=dict(Portland=7, Fairfield=21),

    ## Distance is measured by drivetime(in minutes).

      Apex=dict(Cary=15, ChapelHill=33, FuquayVarina=21, HollySprings=12, Raleigh=21),
      Cary=dict(Apex=15, ChapelHill=31, Durham=23,Raleigh=31),
      ChapelHill=dict(Apex=33, Carrboro=6, Cary=31, Durham=24),
      Clayton=dict(Knightdale=20, Raleigh=26, WakeForest=43, Wendell=19),
      Durham=dict(Cary=23, ChapelHill=24, Raleigh=17, WakeForest=37),
      FuquayVarina=dict(Apex=21, HollySprings=10, Raleigh=32),
      Garner=dict(HollySprings=27, Raleigh=13),
      HollySprings=dict(Apex=12, FuquayVarina=10, Garner=27),
      Knightdale=dict(Clayton=20, Raleigh=18, Wendell=12),
      Raleigh=dict(Apex=21, Cary=17, Clayton=26, Durham=31, FuquayVarina=32, Garner=13, Knightdale=18, WakeForest=29),
      WakeForest=dict(Clayton=43, Durham=37, Raleigh=29, Youngsville=7),
      Wendell=dict(Clayton=19, Knightdale=12)))
      ## (x,y)
      #(Longitude, Latitude) coordinates
triangle_map.locations = dict(
            Apex=(78.8503,35.7327), Cary=(78.7811,35.7915), Carrboro=(79.0753,35.9101),
            ChapellHill=(79.0558,35.9132), Clayton=(78.4564,35.6507), Durham=(78.8986,35.9940),
            FuquayVarina=(78.8000,35.5843), Garner=(78.6142,35.7113), HollySprings=(78.8336,35.6513),
            Knightdale=(78.4806,35.7877), Raleigh=(78.6382,35.7796), WakeForest=(78.5097, 35.9799),
            Wendell=(78.3697,35.7810), Youngsville=(78.4744,36.0249))
# very rough coordinates created by laying out the map onto an x,y grid
# triangle_map.locations = dict(
#               Apex=(27,72), Cary=(45,55), Carrboro=(7,20),
#               ChapellHill=(5,20), Clayton=(105,79), Durham=(34,7),
#               FuquayVarina=(40,99), Garner=(83,75), HollySprings=(45,80),
#               Knightdale=(90,60), Raleigh=(70,63), WakeForest=(108,8),
#               Wendell=(115,63), Youngsville=(93,4))

triangle_puzzle_RD = search.GraphProblem('Raleigh', 'Durham',  triangle_map)
triangle_puzzle_RC = search.GraphProblem('Raleigh', 'ChapelHill', triangle_map) # BFS is better than DFS
triangle_puzzle_CC = search.GraphProblem('ChapelHill', 'Clayton', triangle_map) # UCS is better than BFS

triangle_puzzle_RD.description = '''
An abbreviated map of the Triangle Area of NC.
This map is unique, to the best of my knowledge.
'''


# This state has a total of three blanks and should have at least 216 possible states
initial = ([['2', '2', '3', '1'],
           ['1', '3', 'x', '2', ],
           ['x', '2', 'x', '2'],
           ['1', '3', '3', '3']])

class oneTwoThree(search.Problem):
    # def __init__(self, initial, goal=None):
    #     """I'm not sure how to build this constructor properly"""
    #
    #
    #     self.initial = ([['2','2','3','1'],
    #                             ['1','3','x','2',],
    #                             ['x','2','x','2'],
    #                             ['1','3','3','3']])
    #     # self.grid_width =
    #
    #     self.goal = ([['2','2','3','1'],
    #                      ['1','3','3','2',],
    #                      ['2','2','1','2'],
    #                      ['1','3','3','3']])

    def actions(self, state):
        One = '1'
        Two = '2'
        Three = '3'
        return state

    def result(self, state, action):
        if action =='1':
            #something should probably happen here...
            return '1'
        elif action == '2':
            #and here...
            return '2'
        elif action == '3':
            #and here...
            return '3'

        else:
            return "Action invalid"

    def goal_test(self, state):
        #there is only one possible way to solve the stated puzzle
        solvedPuzzle = ([['2','2','3','1'],
                        ['1','3','3','2',],
                        ['2','2','1','2'],
                        ['1','3','3','3']])

        return state == solvedPuzzle
    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1



oneTwoThree_puzzle = oneTwoThree(initial)

oneTwoThree_puzzle.label = '123 Puzzle'

myPuzzles = [
    triangle_puzzle_RD,
    triangle_puzzle_RC,
    triangle_puzzle_CC,
    # oneTwoThree_puzzle


]