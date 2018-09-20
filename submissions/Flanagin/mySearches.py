import search
from math import(cos, pi)


# A sample map problem
sumner_map = search.UndirectedGraph(dict(
   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   Cottontown=dict(Portland=18),
   Fairfield=dict(Mitchellville=21, Portland=17),
   Mitchellville=dict(Portland=7, Fairfield=21),
))

# Nashville, Atlanta, College Station, Baltimore, Raleigh, St. Louis, Gainsville,

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner'
sumner_puzzle.description = '''
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


la_map = search.UndirectedGraph({
    'Beverly Hills': {'Hollywood': 15, 'Santa Monica': 15},
    'Calabasas': {'Westlake Village': 15, 'Woodland Hills': 11},
    'Disneyland': {'Downtown': 28, 'Venice Beach': 51},
    'Downtown': {'Disneyland': 28, 'Hollywood': 16, 'Santa Monica': 20},
    'Hollywood': {'Beverly Hills': 15, 'Downtown': 16, 'Woodland Hills': 24},
    'Malibu': {'Santa Monica': 33, 'Westlake Village': 26},
    'Santa Monica': {'Beverly Hills': 15, 'Downtown': 20, 'Malibu': 33, 'Venice Beach': 8},
    'Venice Beach': {'Disneyland': 51, 'Santa Monica': 8, 'Woodland Hills': 32},
    'Westlake Village': {'Calabasas': 15, 'Malibu': 26},
    'Woodland Hills': {'Calabasas': 11, 'Hollywood': 24, 'Venice Beach': 32}
})

la_puzzle = search.GraphProblem('Woodland Hills', 'Disneyland', la_map)
la_puzzle.label = 'Los Angeles'

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


# https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/slant.html#5x5:b0a10a2f112b21a2b2b30a01c
class Slant(search.Problem):
    # s1 = slant((slashes, constraints), empty)

    # don't think about how to solve the problem
    # think about how to represent arbitrary states

    def __init__(self, board_height, board_width, constraints):
        board = []
        for i in range(0, board_height):
            row = []
            for j in range(0, board_width):
                row.append('_')
            board.append(row)
        num_constraints = 0
        for m in range(len(constraints)):
            for n in range(len(constraints[0])):
                if constraints[m][n] != '.':
                    num_constraints = num_constraints + 1
        self.constraints = constraints
        self.num_constraints = num_constraints
        self.state = board
        self.initial = board

    def actions(self, state):
        return ['\\', '/', '_']

    def result(self, state, action):
        if state == '_':
            return '\\'
        elif state == '\\':
            return '/'
        else:
            return '\\'

    def goal_test(self, state):
        # needs to check that all corner constraints are met
        done = False
        board = self.state
        constraints = self.constraints
        constraints_passed = 0
        for i in range(len(constraints)):
            for j in range(len(constraints[0])):
                if constraints[i][j] != '.':
                    count = 0
                    try:
                        if board[i-2][j-1] == '\\':
                            count = count + 1
                    except:
                        pass
                    try:
                        if board[i-2][j] == '/':
                            count = count + 1
                    except:
                        pass
                    try:
                        if board[i][j-1] == '/':
                            count = count + 1
                    except:
                        pass
                    try:
                        if board[i][j] == '\\':
                            count = count + 1
                    except:
                        pass
                    count_string = str(count)
                    if count_string == constraints[i][j]:
                        constraints_passed = constraints_passed + 1
        if constraints_passed == self.num_constraints:
            done = True
        # needs to check that there are no cycles
        else:
            done = False
        return done

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


# trivial
slant_constraints1 = [['1', '.'], ['.', '.']]
slant_puzzle1 = Slant(1, 1, slant_constraints1)
slant_puzzle1.label = 'Slant 1'

'''
# https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/slant.html#5x5:b0a10a2f112b21a2b2b30a01c
slant_constraints2 = [['.', '.', '0', '.', '1', '0'],
               ['.', '2', '.', '.', '.', '.'],
               ['.', '.', '1', '1', '2', '.'],
               ['.', '2', '1', '.', '2', '.'],
               ['.', '2', '.', '.', '3', '0'],
               ['.', '0', '1', '.', '.', '.']]
slant_puzzle2 = Slant(5, 5, slant_constraints2)
slant_puzzle2.label = 'Slant 2' '''

# swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'


mySearches = [
 #   swiss_puzzle,
 #   sumner_puzzle,
 #  romania_puzzle,
    switch_puzzle,
    la_puzzle,
    slant_puzzle1
]

mySearchMethods = []
