import search
from math import(cos, pi)

mediterranean_map = search.UndirectedGraph(dict(
    Alexandria=dict(Rome=53, Byzantium=18, Crete=12, Cyprus=6.5, Cyrene=8, Massilia=58, Myra=6, Naples=60, Rhodes=8),
    Ascalon=dict(Thessalonica=15),
    Berytus=dict(Rhodes=8),
    Byzantium=dict(Alexandria=9, Gaza=5, Rhodes=10),
    Caesarea=dict(Rhodes=10),
    Carthage=dict(Gibraltar=7, Rome=3),
    Corinth=dict(Naples=5),
    Crete=dict(Alexandria=3.5, Cyrene=1.5),
    Cyprus=dict(Alexandria=2, Rhodes=4.5),
    Cyrene=dict(Alexandria=4.5, Crete=2),
    Epidamnus=dict(Rome=5),
    Gaza=dict(Byzantium=5, Rhodes=11),
    Gibraltar=dict(Rome=8, Carthage=7),
    Massilia=dict(Rome=2.5, Alexandria=25),
    Myra=dict(Alexandria=3),
    Naples=dict(Corinth=7, Alexandria=10, Rome=3),
    Narbo=dict(Utica=5, Rome=3),
    Rhodes=dict(Alexandria=3.5, Berytus=3.5, Byzantium=10, Caesarea=3.5, Cyprus=2, Gaza=3.5, Rome=52, Tyre=4),
    Rome=dict(Alexandria=11, Carthage=3, Epidamnus=5, Gibraltar=7, Massilia=5, Naples=1, Narbo=3, Rhodes=9, Tarraco=4),
    Tarraco=dict(Rome=4),
    Thessalonica=dict(Ascalon=12),
    Tyre=dict(Rhodes=10),
))
mediterranean_map.locations = dict(
    Alexandria=(31.2, 29.91),
    Ascalon=(31.67, 34.57),
    Berytus=(0, 0),
    Byzantium=(0, 0),
    Caesarea=(0, 0),
    Carthage=(0, 0),
    Corinth=(0, 0),
    Crete=(0, 0),
    Cyprus=(0, 0),
    Cyrene=(0, 0),
    Epidamnus=(0, 0),
    Gaza=(0, 0),
    Gibraltar=(0, 0),
    Massilia=(0, 0),
    Myra=(0, 0),
    Naples=(0, 0),
    Narbo=(0, 0),
    Rhodes=(0, 0),
    Rome=(0, 0),
    Tarraco=(0, 0),
    Thessalonica=(0, 0),
    Tyre=(0, 0),
)

mediterranean_puzzle = search.GraphProblem('Gibraltar', 'Alexandria', mediterranean_map)

mediterranean_puzzle.description = '''
An abbreviated map of ports in the Mediterranean Sea.
Times are based from the article "Speed Under Sail of Ancient Ships", a journal released by the University of Chicago.
http://penelope.uchicago.edu/Thayer/E/Journals/TAPA/82/Speed_under_Sail_of_Ancient_Ships*.html
'''

class ColorMaze(search.Problem):
    colorCounts = dict()

    def getPosition(self, state):
        for row in range(0, len(state)):
            for col in range(0, len(state[row])):
                if state[row][col] == '*':
                    return (row, col)

    def setPositionValue(self, grid, coor, value):
        r,c = coor
        for row in range(0, len(grid)):
            for col in range(0, len(grid[row])):
                if r == row and c == col:
                    grid[row][col] = value
                    return grid

    def actions(self, state):
        return ['u', 'd', 'l', 'r']

    def result(self, state, action):

        r, c = self.getPosition(state)
        newState = state

        if action == 'u':
            r -= 1
        elif action == 'd':
            r += 1
        elif action == 'l':
            c -= 1
        elif action == 'r':
            c += 1


        # Keep within bounds of grid
        if r < 0 or r >= len(state): return state
        if c < 0 or r >= len(state[0]): return state

        for row in range(0, len(newState)):
            for col in range(0, len(row)):
                if r == row and c == col:
                    if newState[row][col] == 'x':
                        # If we're at the same spot
                        return state

                    self.colorCounts[newState[row][col]] += 1
                    newState[row][col] = '*'
                    # Mark where we've been
                    self.setPositionValue(newState, (r,c), 'x')
                    break

        return newState


    def goal_test(self, state):
        position = self.getPosition(state)
        # End in upper right corner
        if position == (0, len(state[0])):
            counts = self.colorCounts.values()
            # Clever way to see if contents are equal to each other
            # Credit to http://stackoverflow.com/a/3844832/4276296
            return len(set(counts)) <= 1

        return False



    def h(self, node):
        state = node.state

        for row in range(0, len(state)):
            for col in range(0, len(row)):
                self.colorCounts[state[row][col]] = 0

        print(self.colorCounts)
        if self.goal_test(state):
            return 0
        else:
            return 1

ColorMaze_puzzle = ColorMaze(
    [
        ['y', '', 'r', ''],
        ['r', 'y', 'r', 'r'],
        ['r', '', 'r', 'y'],
        ['*', 'r', 'r', 'y'],
    ]
)
ColorMaze_puzzle.label = 'Color Maze'

myPuzzles = [
    # One of these is usually has UCS over BFS
    mediterranean_puzzle,
    search.GraphProblem('Tyre', 'Utica', mediterranean_map),
    # BFS is better than DFS
    search.GraphProblem('Massilia', 'Rhodes', mediterranean_map),
    ColorMaze_puzzle,
]