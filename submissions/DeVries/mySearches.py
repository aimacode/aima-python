import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    # Cottontown=dict(Portland=18),
#    # Fairfield=dict(Mitchellville=21, Portland=17),
#    # Mitchellville=dict(Portland=7, Fairfield=21),
#     Sioux_Falls=dict(Tea=19,Hartford=21),
#     Hartford=dict(Sioux_Falls=21,Humboldt=10),
#     Humboldt=dict(Hartford=10,Montrose=8),
#     Montrose=dict(Humboldt=8,Alexandria=31),
#     Alexandria=dict(Montrose=31,Mt_Vernon=27),
#     Mt_Vernon=dict(Alexandria=27,Plankinton=16),
#     Plankinton=dict(Mt_Vernon=16,Corsica=26),
#     Corsica=dict(Plankinton=26,Armour=14),
#     Tea=dict(Sioux_Falls=19,Kaylor=73,Menno=56),
#     Menno=dict(Tea=56,Tripp=22),
#     Tripp=dict(Menno=22,Armour=28),
#     Kaylor=dict(Tea=73,Armour=38),
#     Armour=dict(Kaylor=38,Tripp=28,Corsica=14)
# ))

sumner_map = search.UndirectedGraph(dict(
   # Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   # Cottontown=dict(Portland=18),
   # Fairfield=dict(Mitchellville=21, Portland=17),
   # Mitchellville=dict(Portland=7, Fairfield=21),
    Sioux_Falls=dict(Tea=147.139,Hartford=227.095),
    Hartford=dict(Sioux_Falls=227.095,Humboldt=132.834),
    Humboldt=dict(Hartford=132.834,Montrose=122.102),
    Montrose=dict(Humboldt=122.102,Alexandria=600.688),
    Alexandria=dict(Montrose=600.688,Mt_Vernon=481.387),
    Mt_Vernon=dict(Alexandria=481.387,Plankinton=225.056),
    Plankinton=dict(Mt_Vernon=225.056,Corsica=300.307),
    Corsica=dict(Plankinton=300.307,Armour=123.167),
    Tea=dict(Sioux_Falls=147.139,Kaylor=1035.65,Menno=770.333),
    Menno=dict(Tea=770.333,Tripp=388.252),
    Tripp=dict(Menno=388.252,Armour=392.186),
    Kaylor=dict(Tea=1035.65,Armour=524.37),
    Armour=dict(Kaylor=524.37,Tripp=392.186,Corsica=123.167)
))

sumner_map.locations = dict(
    Sioux_Falls=(3547, 6728), Hartford=(3623, 6942), Humboldt=(3645, 7073),
    Montrose=(3698, 7183), Alexandria=(3653, 7782), Mt_Vernon=(3710, 8260),
    Plankinton=(3715, 8485), Corsica=(3425, 8407), Tea=(3446, 6835),
    Menno=(3239, 7577), Tripp=(3225, 7965), Kaylor=(3188, 7838),
    Armour=(3318, 8346))

sumner_puzzle = search.GraphProblem('Sioux_Falls', 'Armour', sumner_map)

sumner_puzzle.label = 'South_Dakota'
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
        state = string2State(state)
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
        state = string2State(state)
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
            temp_state = state2String(temp_state)
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
            temp_state = state2String(temp_state)
            return temp_state
        elif action == 'left':
            temp_state[1][1] = state[1][1] - 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][0]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][0] = state[2][4]
            temp_state[2][1] = state[2][5]
            temp_state[2][4] = state[2][1]
            # temp_state[2][5] = state[2][2]
            temp_state = state2String(temp_state)
            return temp_state
        elif action == 'right':
            temp_state[1][1] = state[1][1] + 1
            new_row = temp_state[1][0]
            new_col = temp_state[1][1]
            temp_state[0][new_row][new_col] = state[2][1]
            temp_state[2][5] = state[0][new_row][new_col]

            temp_state[2][0] = state[2][5]
            temp_state[2][1] = state[2][4]
            temp_state[2][4] = state[2][0]
            # temp_state[2][5] = state[2][2]
            temp_state = state2String(temp_state)
            return temp_state
        else:
            state = state2String(state)
            return state

    def goal_test(self, state):
        state = string2State(state)
        for row in state[0]:
            for col in row:
                if col == True:
                    return False
        return True

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1

def state2String(myState):
    myString = ""
    for row in myState[0]:
        for col in row:
            myString += str(col) + ","
        myString = myString[:-1]
        myString += "\n"
    myString = myString[:-1]
    myString += "|"
    myString += str(myState[1][0]) + "," + str(myState[1][1]) + "|"
    for cubeSide in myState[2]:
        myString += str(cubeSide) + ","
    myString = myString[:-1]
    return myString

def string2State(myString):
    myState = []
    splitString = myString.split('|')
    rows = splitString[0].split('\n')
    newRows = []
    for row in rows:
        newRows.append(row.split(','))
    myState.append(newRows)
    # CHANGE VALUES IN EACH LIST TO THEIR PROPER TYPE
    coords = splitString[1].split(',')
    myState.append(coords)

    cube = splitString[2].split(',')
    myState.append(cube)

    rowIndex = 0
    for row in myState[0]:
        colIndex = 0
        for col in row:
            myState[0][rowIndex][colIndex] = bool(myState[0][rowIndex][colIndex] in ['True'])
            colIndex += 1
        rowIndex += 1

    myState[1][0] = int(myState[1][0])
    myState[1][1] = int(myState[1][1])

    cubeIndex = 0
    for item in myState[2]:
        myState[2][cubeIndex] = bool(myState[2][cubeIndex] in ['True'])
        cubeIndex += 1



    return myState


#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
# uniform_cost_search yields a solution 4+ moves deep (6 moves), and BFS yields a better solution than DFS.
initial_state =  [[[True,False,False],[False,True,True],[False,True,False]], [0,1], [False,False,False,False,False,False]]
initial_state_string = state2String(initial_state)
switch_puzzle = LightSwitch(initial_state_string)
switch_puzzle.label = 'Cube'
# uniform_cost_search yields a solution 1 move deep.
initial_state =  [[[True, False],[False,False]], [0,1], [False,False,False,False,False,False]]
initial_state_string = state2String(initial_state)
switch_puzzle2 = LightSwitch(initial_state_string)
switch_puzzle2.label = 'Cube2'

# uniform_cost_search yields a solution 2 moves deep.
initial_state =  [[[True,True,False],[False,False,False]], [0,1], [False,False,False,False,False,False]]
initial_state_string = state2String(initial_state)
switch_puzzle3 = LightSwitch(initial_state_string)
switch_puzzle3.label = 'Cube3'

#
# initial_state =  [[[True,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[False,False,False,False,False,False],[True,False,False,False,False,True]], [3,3], [False,False,False,False,False,False]]
# initial_state_string = state2String(initial_state)
# switch_puzzle4 = LightSwitch(initial_state_string)
# switch_puzzle4.label = 'Cube4'

# initial_state =  [[[True,False,False],[False,True,True],[False,True,False]], [0,1], [False,False,False,False,False,False]]
# initial_state =  [[[True,False,False],[False,True,True],[False,True,False]], [0,1], [False,False,False,False,False,False]]


# initial_state_string = state2String(initial_state)
# switch_puzzle = LightSwitch(initial_state_string)
# # switch_puzzle = LightSwitch('on')
# switch_puzzle.label = 'Cube'

# testing stmts
# print(state2String([[[True, False],[False,True]], [0,1], [False,False,False,False,False,False]]))
# print(string2State(state2String([[[True, False],[False,True]], [0,1], [False,False,False,False,False,False]])))

mySearches = [
 #   swiss_puzzle,
    sumner_puzzle,
    # romania_puzzle,
    switch_puzzle,
    switch_puzzle2,
    switch_puzzle3
    # switch_puzzle4
]
mySearchMethods = []
