import search
from math import(cos, pi)

# A sample map problem
uk_map = search.UndirectedGraph(dict(
   Birmingham=dict(Oxford=68, London=117, Wolverhampton=23,Cheltenham=67),
   Oxford=dict(London=84, Birmingham=68),
   London=dict(Birmingham=117, Oxford=84, Luton=104),
   Coventry=dict(Birmingham = 43, Luton= 204),
   Luton=dict(Coventry=204, london=104),
   Wolverhampton=dict(Birmingham=28),
   Cheltenham=dict(Birmingham=67),
   Buckingham=dict(Birmingham=101,Aylesbury=45,),
   Aylesbury=dict(Buckingham = 45, Watford =81 ),
   Watford=dict(Aylesbury = 81, London = 20)

))
uk_map.locations= dict(
    Birmingham=(52.4862, 1.8904),
    Oxford=(51.7520, 1.2577),
    London=(51.5074, 0.1278),
    Coventry=(52.4068, 1.5197),
    Luton=(51.8787, 0.4200),
    Wolverhampton=(52.5870, 2.1288),
    Cheltenham=(51.8994, 2.0783),
    Buckingham=(51.5014, 0.1419),
    Aylesbury=(51.8156, 0.808),
    Watford=(51.6565,0.3903))

uk_puzzle = search.GraphProblem('Birmingham', 'Watford', uk_map)

uk_puzzle.label = 'UK'
uk_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

romania_map = search.UndirectedGraph(dict(
    A=dict(Z=75,S=140,T=118),
    Z=dict(O=71,A=75),
    S=dict(O=151,R=80,F=99),
    T=dict(A=118,L=111),
    O=dict(Z=71,S=151),
    L=dict(T=111,M=70),
    M=dict(L=70,D=75),
    D=dict(M=75,C=120),
    R=dict(S=80,C=146,P=97),
    C=dict(R=146,P=138,D=120),
    F=dict(S=99,B=211),
    P=dict(R=97,C=138,B=101),
    B=dict(G=90,P=101,F=211),
))

romania_puzzle = search.GraphProblem('A', 'B', romania_map)

romania_puzzle.label = 'Romania'
romania_puzzle.description = '''
The simplified map of Romania, per
Russall & Norvig, 3rd Ed., p. 68.
'''

# A trivial Problem definition
class LightSwitch(search.Problem):
    def actions(self, state):
        return ['up', 'down']

    def result(self, state, action):
        if action == 'up':
            return 'on'
        else:
            return 'off'

    def goal_test(self, state):
        return state == 'on'

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1



#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

# The sliding tile problem beloved of AI teachers,
# because it has a decent heuristic.
class SlidingTile(search.Problem):
    # initial and goal states are strings of the form
    #  '1,2,3,4|5,6,7,8|9,10,11,_'
    #  Where _ marks the missing tile.
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

#tile_puzzle = SlidingTile('1,2,4|5,_,3', '5,4,3|2,1,_')
tile_puzzle = SlidingTile('1,2,3|5,_,3', '1,2,3|4,5,_')
tile_puzzle.label = '4x4 Tiles'



mySearches = [
 #   swiss_puzzle,
    uk_puzzle,
    romania_puzzle,
    switch_puzzle,


]
mySearchMethods = []