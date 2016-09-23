import search
from math import(cos, pi)

# scale latitude and longitude,
# so that the distance between two points is an estimate of
# the minutes it takes to drive between them
def travelTime(lat, long):
    latFudge = 43 # convert degrees of latitude into travel minutes
    longFudge = latFudge * cos(long * pi / 180)
    return lat*latFudge, long*longFudge

sumner_map = search.UndirectedGraph(dict(
    Portland=dict(Mitchellville=7, Fairfield=17, Graball=18,
                  Cottontown=18),
    Cottontown=dict(Portland=18, Graball=14, S=6),
    Fairfield=dict(Graball=16,Mitchellville=21, Portland=17),
    Graball=dict(Cottontown=14, Fairfield=17, Portland=18),
    Mitchellville=dict(Portland=7, Fairfield=21, Q=6),
    Q=dict(Mitchellville=6, R=6),
    R=dict(Q=6, S=6),
    S=dict(R=6, Cottontown=6),
))
sumner_map.locations = dict(
    Portland=travelTime(36.581538,-86.6534912),
    Fairfield=travelTime(36.6182314,-86.3761194),
    Cottontown=travelTime(36.449575,-86.5697851),
    Graball=travelTime(36.4788454,-86.4750586),
    Mitchellville=travelTime(36.6334132,-86.5480556),
    Q=travelTime(36.49553455,-86.564352725),
    R=travelTime(36.5414941,-86.55892035),
    S=travelTime(36.58745365,-86.553487975),
)

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
sumner_puzzle.label = 'Sumner Map'

# The sliding tile problem beloved of AI teachers,
# because it has a decent heuristic.
class SlidingTile(search.Problem):

    # initial and goal states are strings of the form
    # '1,2,3,4|5,6,7,8|9,10,11,_'
    # Where _ marks the missing tile.
    def __init__(self, initial, goal=None):
        super().__init__(initial, goal)
        iGrid = self.state2grid(initial)
        self.rows = len(iGrid)
        self.cols = len(iGrid[0])
        if not goal:
            gGrid = [[0 for c in range(self.cols)] for r in range(self.rows)]
            count = self.rows * self.cols
            list = self.state2list(initial)
            goalList = sorted(list)
            for r in range(self.rows):
                for c in range(self.cols):
                    count -= 1
                    gGrid[r][c] = goalList[count]
            self.goal = self.grid2state(gGrid)

    def grid2state(self, grid):
        rString = ''
        rowSeparator = ''
        for row in grid:
            rString += rowSeparator
            cellSeparator = ''
            for cell in row:
                rString += cellSeparator + cell
                cellSeparator = ','
            rowSeparator = '|'
        return rString

    def state2grid(self, state):
        grid = []
        rows = state.split('|')
        for row in rows:
            cells = row.split(',')
            grid.append(cells)
        return grid

    def state2list(self, state):
        list = []
        rows = state.split('|')
        for row in rows:
            cells = row.split(',')
            for cell in cells:
                list.append(cell)
        return list

    def grid2dict(self, grid):
        locations = {}
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                cell = grid[r][c]
                locations[cell] = (r, c)
        return locations

    def state2dict(self, state):
        grid = self.state2grid(state)
        return self.grid2dict(grid)

    # directions you can slide a tile
    def actions(self, state):
        dict = self.state2dict(state)
        r, c = dict['_']
        moves = []
        if r > 0:
            moves.append('d')
        if r < self.rows - 1:
            moves.append('u')
        if c > 0:
            moves.append('l')
        if c < self.cols - 1:
            moves.append('r')
        return moves

    def result(self, state, action):
        grid = self.state2grid(state)
        dict = self.grid2dict(grid)
        r1, c1 = dict['_']
        if action == 'd':
            r2 = r1 - 1
            c2 = c1
        if action == 'u':
            r2 = r1 + 1
            c2 = c1
        if action == 'l':
            r2 = r1
            c2 = c1 - 1
        if action == 'r':
            r2 = r1
            c2 = c1 + 1
        if r2 < 0:
            raise IndexError
        if r2 >= self.rows:
            raise IndexError
        if c2 < 0:
            raise IndexError
        if c2 >= self.cols:
            raise IndexError
        grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
        state = self.grid2state(grid)
        return state

    def manhattan(self, s1, s2):
        dict1 = self.state2dict(s1)
        dict2 = self.state2dict(s2)
        sum = 0
        for k in dict1.keys():
            r1, c1 = dict1[k]
            r2, c2 = dict2[k]
            sum += abs(r2 - r1)
            sum += abs(c2 - c1)
        return sum

    def goal_test(self, state):
        return state == self.goal

    def h(self, node):
        state = node.state
        goal = self.goal
        return self.manhattan(state, goal)

    def prettyPrint(self, state):
        output = ''
        rowSeparator = ''
        for row in state.split('|'):
            output += rowSeparator
            cellSeparator = ''
            for cell in row.split(','):
                output += cellSeparator
                output += cell
                cellSeparator = ' '
            rowSeparator = '\n'
        return output

tile_puzzle = SlidingTile('1,2,4|5,_,3', '5,4,3|2,1,_')
tile_puzzle.label = '3x2 Tiles'

myPuzzles = [
    sumner_puzzle,
    tile_puzzle
]