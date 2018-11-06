import search
from math import(cos, pi)

# A sample map problem
#sumner_map = search.UndirectedGraph(dict(
#   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#   Cottontown=dict(Portland=18),
#   Fairfield=dict(Mitchellville=21, Portland=17),
#   Mitchellville=dict(Portland=7, Fairfield=21),
#))

fayette_map = search.UndirectedGraph(dict(
   Uniontown=dict(Brownsville=15, Connellsville=12, Farmington=12, Perryopolis = 16, Smithfield = 9),
   Brownsville=dict(Uniontown = 15, Perryopolis = 10),
   Perryopolis=dict(Brownsville = 10, Connellsville = 13, Uniontown = 16),
   Connellsville=dict(Perryopolis=12, Ohiopyle=19, Uniontown = 12, Normalville = 8),
   Ohiopyle=dict(Farmington=7, Connellsville=19, Normalville =10, Markleysburg = 9),
   Farmington=dict(Ohiopyle=7, Uniontown=12, Markleysburg = 9),
   Smithfield=dict(Uniontown=9),
   Normalville=dict(Connellsville=8, Ohiopyle = 10),
   Markleysburg=dict(Farmington = 9, Ohiopyle =13)
))

fayette_map.locations = dict(
    Uniontown =(39.8973431,-79.742057),
    Brownsville=(40.0187766,-79.9103586),
    Perryopolis=(40.086682,-79.76838),
    Connellsville=(40.0147711,-79.6209733),
    Ohiopyle=(39.8687992,-79.5033132),
    Farmington = (39.8072964,-79.5831068),
    Smithfield = (39.8013187,-79.827878),
    Normalville = (39.9986836,-79.4656011),
    Markleysburg =(39.7368018,-79.458878)
                )



#sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

#sumner_puzzle.label = 'Sumner'
#sumner_puzzle.description = '''
#An abbreviated map of Sumner County, TN.
#This map is unique, to the best of my knowledge.
#'''

fayette_puzzle = search.GraphProblem('Perryopolis', 'Markleysburg', fayette_map)
fayette_puzzle.label = 'Fayette P2M'
fayette_puzzle.description = '''
An abbreviated map of Fayette County, PA.
This map is unique, to the best of my knowledge.
'''

# added by whh
fp2 = search.GraphProblem('Markleysburg', 'Connellsville', fayette_map)
fp2.label = 'Fayette M2C'

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


# This class solves puzzles created for the Rush Hour Board Game
class RushHourEasy(search.Problem):

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
            grid = self.state2grid(state)
            dict = self.state2dict(state)
            del dict['__']
            rY1, cY1 = dict['Y1']

            rO1, cO1 = dict['O1']
            rR1, cR1 = dict['R1']

            moves = []

            Yd = self.rows
            Yu = 0
            Ol = 0
            Or = self.cols
            Rl = 0
            Rr = self.cols

            if rY1 + 3 <= self.rows:
                Yd = rY1 + 3
            if rY1 - 1 >= 0:
                Yu = rY1 - 1
            if cO1 - 1 >= 0:
                Ol = cO1 - 1
            if cO1 + 2 <= self.cols:
                Or = cO1 + 2
            if cR1 - 1 >= 0:
                Rl = cR1 - 1
            if cR1 + 2 <= self.cols:
                Rr = cR1 + 2

            if rY1 < self.rows - 3 and ((grid[Yd][cY1]) not in dict):
                moves.append('dY')
            if rY1 > 0 and ((grid[Yu][cY1]) not in dict):
                moves.append('uY')
            if cO1 > 0 and ((grid[rO1][Ol]) not in dict):
                moves.append('lO')
            if cO1 < self.cols - 2 and ((grid[rO1][Or]) not in dict):
                moves.append('rO')
            if cR1 > 0 and ((grid[rR1][Rl]) not in dict):
                moves.append('lR')
            if cR1 < self.cols - 2 and ((grid[rR1][Rr]) not in dict):
                moves.append('rR')
            return moves

        def result(self, state, action):
            grid = self.state2grid(state)
            dict = self.grid2dict(grid)
            rY1, cY1 = dict['Y1']

            rO1, cO1 = dict['O1']
            rR1, cR1 = dict['R1']
            rY2, cY2 = dict['Y2']

            rO2, cO2 = dict['O2']
            rR2, cR2 = dict['R2']
            rY3, cY3 = dict['Y3']

            if action == 'uY':
                r1 = rY1 - 1
                c1 = cY1
                grid[rY1][cY1], grid[r1][c1] = grid[r1][c1], grid[rY1][cY1]
                r2 = rY2 - 1
                c2 = cY2
                grid[rY2][cY2], grid[r2][c2] = grid[r2][c2], grid[rY2][cY2]
                r3 = rY3 - 1
                c3 = cY3
                grid[rY3][cY3], grid[r3][c3] = grid[r3][c3], grid[rY3][cY3]
            if action == 'dY':
                r1 = rY3 + 1
                c1 = cY3
                grid[rY3][cY3], grid[r1][c1] = grid[r1][c1], grid[rY3][cY3]
                r2 = rY2 + 1
                c2 = cY2
                grid[rY2][cY2], grid[r2][c2] = grid[r2][c2], grid[rY2][cY2]
                r3 = rY1 + 1
                c3 = cY1
                grid[rY1][cY1], grid[r3][c3] = grid[r3][c3], grid[rY1][cY1]
            if action == 'lO':
                r1 = rO1
                c1 = cO1 - 1
                grid[rO1][cO1], grid[r1][c1] = grid[r1][c1], grid[rO1][cO1]
                r2 = rO2
                c2 = cO2 - 1
                grid[rO2][cO2], grid[r2][c2] = grid[r2][c2], grid[rO2][cO2]
            if action == 'rO)':
                r1 = rO2
                c1 = cO2 + 1
                grid[rO2][cO2], grid[r1][c1] = grid[r1][c1], grid[rO2][cO2]
                r2 = rO1
                c2 = cO1 + 1
                grid[rO1][cO1], grid[r2][c2] = grid[r2][c2], grid[rO1][cO1]
            if action == 'lR':
                r1 = rR1
                c1 = cR1 - 1
                grid[rR1][cR1], grid[r1][c1] = grid[r1][c1], grid[rR1][cR1]
                r2 = rR2
                c2 = cR2 - 1
                grid[rR2][cR2], grid[r2][c2] = grid[r2][c2], grid[rR2][cR2]
            if action == 'rR':
                r1 = rR2
                c1 = cR2 + 1
                grid[rR2][cR2], grid[r1][c1] = grid[r1][c1], grid[rR2][cR2]
                r2 = rR1
                c2 = cR1 + 1
                grid[rR1][cR1], grid[r2][c2] = grid[r2][c2], grid[rR1][cR1]

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


# This class solves puzzles created for the Rush Hour Board Game
class RushHour(search.Problem):

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
            grid = self.state2grid(state)
            dict = self.state2dict(state)
            del dict['__']
            rY1, cY1 = dict['Y1']
            rG1, cG1 = dict['G1']
            rO1, cO1 = dict['O1']
            rR1, cR1 = dict['R1']

            moves = []

            Yd = self.rows
            Yu = 0
            Gd = self.rows
            Gu = 0
            Ol = 0
            Or = self.cols
            Rl = 0
            Rr = self.cols

            if rY1 + 3 <= self.rows:
                Yd = rY1 + 3
            if rY1 - 1 >= 0:
                Yu = rY1 - 1
            if rG1 + 2 <= self.rows:
                Gd = rG1 + 2
            if rG1 - 1 >= 0:
                Gu = rG1 - 1
            if cO1 - 1 >= 0:
                Ol = cO1 - 1
            if cO1 + 2 <= self.cols:
                Or = cO1 + 2
            if cR1 - 1 >= 0:
                Rl = cR1 - 1
            if cR1 + 2 <= self.cols:
                Rr = cR1 + 2

            if rY1 < self.rows - 3 and ((grid[Yd][cY1]) not in dict):
                moves.append('dY')
            if rY1 > 0 and ((grid[Yu][cY1]) not in dict):
                moves.append('uY')
            if rG1 < self.rows - 2 and ((grid[Gd][cG1]) not in dict):
                moves.append('dG')
            if rG1 > 0 and ((grid[Gu][cG1]) not in dict):
                moves.append('uG')
            if cO1 > 0 and ((grid[rO1][Ol]) not in dict):
                moves.append('lO')
            if cO1 < self.cols - 2 and ((grid[rO1][Or]) not in dict):
                moves.append('rO')
            if cR1 > 0 and ((grid[rR1][Rl]) not in dict):
                moves.append('lR')
            if cR1 < self.cols - 2 and ((grid[rR1][Rr]) not in dict):
                moves.append('rR')
            return moves

        def result(self, state, action):
            grid = self.state2grid(state)
            dict = self.grid2dict(grid)
            rY1, cY1 = dict['Y1']
            rG1, cG1 = dict['G1']
            rO1, cO1 = dict['O1']
            rR1, cR1 = dict['R1']
            rY2, cY2 = dict['Y2']
            rG2, cG2 = dict['G2']
            rO2, cO2 = dict['O2']
            rR2, cR2 = dict['R2']
            rY3, cY3 = dict['Y3']

            if action == 'uY':
                r1 = rY1 - 1
                c1 = cY1
                grid[rY1][cY1], grid[r1][c1] = grid[r1][c1], grid[rY1][cY1]
                r2 = rY2 - 1
                c2 = cY2
                grid[rY2][cY2], grid[r2][c2] = grid[r2][c2], grid[rY2][cY2]
                r3 = rY3 - 1
                c3 = cY3
                grid[rY3][cY3], grid[r3][c3] = grid[r3][c3], grid[rY3][cY3]
            if action == 'dY':
                r1 = rY3 + 1
                c1 = cY3
                grid[rY3][cY3], grid[r1][c1] = grid[r1][c1], grid[rY3][cY3]
                r2 = rY2 + 1
                c2 = cY2
                grid[rY2][cY2], grid[r2][c2] = grid[r2][c2], grid[rY2][cY2]
                r3 = rY1 + 1
                c3 = cY1
                grid[rY1][cY1], grid[r3][c3] = grid[r3][c3], grid[rY1][cY1]
            if action == 'uG':
                r1 = rG1 - 1
                c1 = cG1
                grid[rG1][cG1], grid[r1][c1] = grid[r1][c1], grid[rG1][cG1]
                r2 = rG2 - 1
                c2 = cG2
                grid[rG2][cG2], grid[r2][c2] = grid[r2][c2], grid[rG2][cG2]
            if action == 'dG':
                r1 = rG2 + 1
                c1 = cG2
                grid[rG2][cG2], grid[r1][c1] = grid[r1][c1], grid[rG2][cG2]
                r2 = rG1 + 1
                c2 = cG1
                grid[rG1][cG1], grid[r2][c2] = grid[r2][c2], grid[rG1][cG1]
            if action == 'lO':
                r1 = rO1
                c1 = cO1 - 1
                grid[rO1][cO1], grid[r1][c1] = grid[r1][c1], grid[rO1][cO1]
                r2 = rO2
                c2 = cO2 - 1
                grid[rO2][cO2], grid[r2][c2] = grid[r2][c2], grid[rO2][cO2]
            if action == 'rO)':
                r1 = rO2
                c1 = cO2 + 1
                grid[rO2][cO2], grid[r1][c1] = grid[r1][c1], grid[rO2][cO2]
                r2 = rO1
                c2 = cO1 + 1
                grid[rO1][cO1], grid[r2][c2] = grid[r2][c2], grid[rO1][cO1]
            if action == 'lR':
                r1 = rR1
                c1 = cR1 - 1
                grid[rR1][cR1], grid[r1][c1] = grid[r1][c1], grid[rR1][cR1]
                r2 = rR2
                c2 = cR2 - 1
                grid[rR2][cR2], grid[r2][c2] = grid[r2][c2], grid[rR2][cR2]
            if action == 'rR':
                r1 = rR2
                c1 = cR2 + 1
                grid[rR2][cR2], grid[r1][c1] = grid[r1][c1], grid[rR2][cR2]
                r2 = rR1
                c2 = cR1 + 1
                grid[rR1][cR1], grid[r2][c2] = grid[r2][c2], grid[rR1][cR1]

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


rush_puzzle = RushHour('__,__,__,__,Y1,__|__,__,__,__,Y2,__|__,__,R1,R2,Y3,__|__,__,__,G1,__,__|__,__,__,G2,O1,O2|__,__,__,__,__,__',
                       '__,__,__,G1,__,__|__,__,__,G2,__,__|__,__,__,__,R1,R2|__,__,__,__,Y1,__|__,__,O1,O2,Y2,__|__,__,__,__,Y3,__')
rush_puzzle.label = '6 x 6, 4 Vehicles'

rush1_puzzle = RushHourEasy('__,__,__,__,Y1,__|__,__,__,__,Y2,__|__,__,R1,R2,Y3,__|__,__,__,__,__,__|__,__,__,__,O1,O2|__,__,__,__,__,__',
                       '__,__,__,__,__,__|__,__,__,__,__,__|__,__,__,__,R1,R2|__,__,__,__,Y1,__|__,__,O1,O2,Y2,__|__,__,__,__,Y3,__')
rush1_puzzle.label = '6 x 6, 3 Vehicles'


#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
    #   swiss_puzzle,
    # sumner_puzzle,
    #romania_puzzle,
    #  switch_puzzle,

    fayette_puzzle,
    fp2,
    rush1_puzzle,
    #rush_puzzle
]

mySearchMethods = []
