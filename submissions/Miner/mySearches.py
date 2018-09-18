import search
from math import(cos, pi)

# A sample map problem
auckland_map = search.UndirectedGraph(dict(
    Auckland=dict(MountAlbert=18, MissionBay=22),
    MountAlbert=dict(Auckland=18, Hillsborough=20, Avondale=7),
    MangereEast=dict(Hillsborough=20, Packuranga=20),
    MissionBay=dict(Auckland=22, Hillsborough=31),
    Hillsborough=dict(MissionBay=31, MangereEast=20, MountAlbert=20, Avondale=13, GlenEden=20),
    Avondale=dict(Hillsborough=13, GlenEden=11, MountAlbert=7),
    GlenEden=dict(Hillsborough=20, Avondale=11),
    Epsom=dict(MountAlbert=15, MissionBay=18, Packuranga=17),
    Packuranga=dict(MissionBay=23, Hillsborough=19, MangereEast=20, Otahuhu=15),
    Otahuhu=dict(Hillsborough=16, Pakuranga=15)
))

auckland_puzzle = search.GraphProblem('Auckland', 'GlenEden', auckland_map)

auckland_puzzle.label = 'Auckland, NZ'
auckland_puzzle.description = '''
# An abbreviated map of Auckland, NZ.x
# This map is unique, to the best of my knowledge.
# '''

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

mySearches = [
 #   swiss_puzzle,
    romania_puzzle,
    switch_puzzle,
    auckland_puzzle
]

mySearchMethods = []
