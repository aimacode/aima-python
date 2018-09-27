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

la_map.locations = {
    'Beverly Hills': (20, 8),
    'Calabasas': (9, 21),
    'Disneyland': (40, 4),
    'Downtown': (26, 16),
    'Hollywood': (23, 19),
    'Malibu': (3, 15),
    'Santa Monica': (16, 14),
    'Venice Beach': (16, 15),
    'Westlake Village': (2, 22),
    'Woodland Hills': (11, 23)
}

la_puzzle = search.GraphProblem('Woodland Hills', 'Disneyland', la_map)
la_puzzle.label = 'Los Angeles 1'

la_puzzle2 = search.GraphProblem('Hollywood', 'Malibu', la_map)
la_puzzle2.label = 'Los Angeles 2'

la_puzzle3 = search.GraphProblem('Calabasas', 'Venice Beach', la_map)
la_puzzle3.label = 'Los Angeles 3'

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

    def __init__(self, constraints):
        board_height = len(constraints)-1
        board_width = len(constraints[0])-1
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
        # self.state = state2string(board)
        self.initial = state2string(board)

    def actions(self, state):
        board = string2state(state)
        actions = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '_':
                    actions.append('\\' + ',' + str(i) + ',' + str(j))
                    actions.append('/' + ',' + str(i) + ',' + str(j))
        return actions

    def result(self, state, action):
        board = string2state(state)
        a, b, c = action.split(',')
        i = int(b)
        j = int(c)
        board[i][j] = a
        return state2string(board)

    def goal_test(self, state):
        board = string2state(state)
        done = False
        # check that every square is not '_'
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '_':
                    return False
        # needs to check that all corner constraints are met
        constraints = self.constraints
        constraints_passed = 0
        for i in range(len(constraints)):
            for j in range(len(constraints[0])):
                count = 0
                if constraints[i][j] != '.':
                    a = i - 1
                    b = j - 1

                    if a >= 0 and b >= 0:
                        try:
                            if board[a][b] == '\\':   # upper left box
                                count = count + 1
                        except:
                            pass
                    if a >= 0:
                        try:
                            if board[a][j] == '/':    # upper right box
                                count = count + 1
                        except:
                            pass
                    if b >= 0:
                        try:
                            if board[i][b] == '/':    # lower left box
                                count = count + 1
                        except:
                            pass
                    try:
                        if board[i][j] == '\\':       # lower right box
                            count = count + 1
                    except:
                        pass
                    '''
                    if a >= 0 and b >= 0:
                        if board[a][b] == '\\':
                            count = count + 1
                        if board[a][j] == '/':
                            count = count + 1
                        if board[i][b] == '/':
                            count = count + 1
                    if board[i][j] == '\\':
                        count = count + 1'''

                    count_string = str(count)
                    if count_string == constraints[i][j]:
                        constraints_passed = constraints_passed + 1
        if constraints_passed == self.num_constraints:
            return True
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


def state2string(state):
    state_string = ''
    for i in range(len(state)):
        for j in range(len(state[0])):
            state_string = state_string + state[i][j] + ','
        state_string = state_string[:-1] + '|'
    state_string = state_string[:-1]
    return state_string


def string2state(state_string):
    state = []
    rows = state_string.split('|')
    for row in rows:
        states = row.split(',')
        state.append(states)
    return state


# trivial
slant_constraints1 = [['1', '.'], ['.', '.']]
slant_puzzle1 = Slant(slant_constraints1)
slant_puzzle1.label = 'Slant 1'

# 2x1
slant_constraints2 = [['.', '2', '.'], ['.', '.', '.']]
slant_puzzle2 = Slant(slant_constraints2)
slant_puzzle2.label = 'Slant 2'

# 2x2
# https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/slant.html#2x2:a2e0a
slant_constraints3 = [['.', '2', '.'],
                      ['.', '.', '.'],
                      ['.', '0', '.']]
slant_puzzle3 = Slant(slant_constraints3)
slant_puzzle3.label = 'Slant 3'

# 3x2
# https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/slant.html#3x2:a2e1a01a
slant_constraints4 = [['.', '2', '.', '.'],
                      ['.', '.', '.', '1'],
                      ['.', '0', '1', '.']]
slant_puzzle4 = Slant(slant_constraints4)
slant_puzzle4.label = 'Slant 4'

# 3x3
# https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/slant.html#3x3:0b011c31d1
slant_constraints5 = [['0', '.', '.', '0'],
                      ['1', '1', '.', '.'],
                      ['.', '3', '1', '.'],
                      ['.', '.', '.', '1']]
slant_puzzle5 = Slant(slant_constraints5)
slant_puzzle5.label = 'Slant 5'

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
    #switch_puzzle,
    la_puzzle,
    la_puzzle2,
    la_puzzle3,
    slant_puzzle1,    # 1x1
    slant_puzzle2,    # 2x1
    slant_puzzle3,    # 2x2
    slant_puzzle4,    # 3x2
    #slant_puzzle5     # 3x3
]

mySearchMethods = []
