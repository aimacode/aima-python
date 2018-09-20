import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
   # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   # Cottontown=dict(Portland=18),
   # Fairfield=dict(Mitchellville=21, Portland=17),
   # Mitchellville=dict(Portland=7, Fairfield=21),
    Sioux_Falls=dict(Tea=19,Hartford=21),
    Hartford=dict(Sioux_Falls=21,Humboldt=10),
    Humboldt=dict(Hartford=10,Montrose=8),
    Montrose=dict(Humboldt=8,Alexandria=31),
    Alexandria=dict(Montrose=31,Mt_Vernon=27),
    Mt_Vernon=dict(Alexandria=27,Plankinton=16),
    Plankinton=dict(Mt_Vernon=16,Corsica=26),
    Corsica=dict(Plankinton=26,Armour=14),
    Tea=dict(Sioux_Falls=19,Kaylor=73,Menno=56),
    Menno=dict(Tea=56,Tripp=22),
    Tripp=dict(Menno=22,Armour=28),
    Kaylor=dict(Tea=73,Armour=38),
    Armour=dict(Kaylor=38,Tripp=28,Corsica=14)
))

sumner_puzzle = search.GraphProblem('Sioux_Falls', 'Armour', sumner_map)

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

# A trivial Problem definition - Copy of the original
# class LightSwitch(search.Problem):
#     def actions(self, state):
#         return ['up', 'down']
#
#     def result(self, state, action):
#         if action == 'up':
#             return 'on'
#         else:
#             return 'off'
#
#     def goal_test(self, state):
#         return state == 'on'
#
#     def h(self, node):
#         state = node.state
#         if self.goal_test(state):
#             return 0
#         else:
#             return 1

# state = [[[booleans for row 1],[booleans for row 2]...], [coords for cube - ex (0,0) = upper left corner], [l,r,f,b,u,d]]
class LightSwitch(search.Problem):
    def actions(self, state):
        action_list = ['up', 'down', 'left', 'right']
        # Need to know if we're at an edge/corner
        if (state[1][0] == 0):
            action_list.remove('up')
        if (state[1][0] == len(state[0])-1):
            action_list.remove('down')
        if (state[1][1] == 0):
            action_list.remove('left')
        if (state[1][1] == len(state[0][0])-1):
            action_list.remove('right')
        return action_list

    def result(self, state, action):
        temp_state = state
        if action == 'up':
            temp_state[1][0] = state[1][0] - 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][2]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][2] = state[2][4]
            temp_state[2][3] = state[2][5]
            temp_state[2][4] = state[2][3]
            # temp_state[2][5] = state[2][2]

            return temp_state
        elif action == 'down':
            temp_state[1][0] = state[1][0] + 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][3]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][2] = state[2][5]
            temp_state[2][3] = state[2][4]
            temp_state[2][4] = state[2][2]
            # temp_state[2][5] = state[2][3]

            return temp_state
        elif action == 'left':
            temp_state[1][0] = state[1][1] - 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][0]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][0] = state[2][4]
            temp_state[2][1] = state[2][5]
            temp_state[2][4] = state[2][1]
            # temp_state[2][5] = state[2][2]

            return temp_state
        elif action == 'right':
            temp_state[1][0] = state[1][1] + 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][1]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][0] = state[2][5]
            temp_state[2][1] = state[2][4]
            temp_state[2][4] = state[2][0]
            # temp_state[2][5] = state[2][2]

            return temp_state
        else:
            return state

    def goal_test(self, state):
        for row in state[0]:
            for col in row:
                if col == False:
                    return False
        return True

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch([[[True, False],[False,True]], [0,1], [False,False,False,False,False,False]])
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    sumner_puzzle,
    romania_puzzle,
    switch_puzzle,
]
mySearchMethods = []
