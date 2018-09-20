import search
from math import(cos, pi)


dallas_map = search.UndirectedGraph (dict(

    Dallas=dict(Rockwall=25, Fortworth = 35),
    Rockwall=dict (Dallas = 25, Richardson = 24),
    Richardson=dict (Rockwall = 24, Coppell = 21),
    Fortworth=dict (Dallas = 35, Azle = 17, Garland = 33, Frisco = 45),
    Azle=dict (Fortworth = 17, Reno = 10, Frisco = 33, Fubu = 41),
    Reno=dict (Azle = 10, Coppell = 47, Garland =19, Lakewood= 45),
    Coppell=dict (Richardson = 21, Reno = 47, Lakewood = 8),
    Frisco=dict (Azle = 33, Fortworth =45),
    Lakewood= dict( Reno = 45, Coppell=8),
    Garland= dict (Fortworth = 33, Reno = 19),
    Fubu= dict(Azle = 41, Nike = 51),
    Nike= dict(Fubu = 51)

))

Dallas_puzzle = search.GraphProblem('Azle', 'Garland', dallas_map)

Dallas_puzzle.label = 'Dallas'
Dallas_puzzle.description = '''
An abbreviated map of Dallas, Tx.
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

# class PushKnob(search.Problem):
#     def __init__(self):
#         self.initState = [(0,0,full),(1,0,empty),(1, 0, empty), (1,1,full) ]
#
#     def actions(self, state):
#         return[x, y, value]
#
#     def result(self, state, action):
#         if (action[0][0], action[0][1], action[1][0], action[1][1])
#             return state
#
#     def goal_test(self, state):
#         return state == [(0,0,empty), (0,1,empty), (1,0,empty), (1,0,empty)]
#
#     def h(self, node):
#         state = node.state
#         if self.goal_test(state):
#             return 0
#         else:
#             return 1



#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

# knob_puzzle = PushKnob('out')
# knob_puzzle.label = 'Push Knob'

mySearches = [
 #   swiss_puzzle,
    Dallas_puzzle,
    romania_puzzle,
    switch_puzzle,
    # knob_puzzle,
]

mySearchMethods = []