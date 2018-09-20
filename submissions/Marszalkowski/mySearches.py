import search
from math import(cos, pi)

# A sample map problem
erie_map = search.UndirectedGraph(dict(
   Clarence=dict(Lancaster=11, Amherst=8),
   Amherst=dict(GrandIsland=15, Clarence=8, Buffalo=19),
   Lancaster=dict(Cheektowaga=4, Clarence=17, WestSeneca=10),
   GrandIsland=dict(Amherst=15, Buffalo=13),
   Cheektowaga=dict(Lancaster=4, Buffalo=10),
   WestSeneca=dict(OrchardPark=4, Buffalo=10),
   OrchardPark=dict(WestSeneca=4, Buffalo=18),
   Buffalo=dict(GrandIsland=13, Amherst=19, Cheektowaga=10, WestSeneca=10, OrchardPark=18),
   Hamburg=dict(Buffalo=12),
   Fredonia=dict(Hamburg=10),
   SBuffalo=dict(Buffalo=3)
))

erie_map.locations = dict(
    Clarence=(27,41), Amherst=(19,41), Lancaster=(27,30), GrandIsland=(4,40),
    Cheektowaga=(23,31), WestSeneca=(24,22), OrchardPark=(26,17), Buffalo=(9,28),
    Hamburg=(20,17), Fredonia=(10,10), SBuffalo=(12,25)
)


erie_puzzle = search.GraphProblem('GrandIsland', 'OrchardPark', erie_map)

erie_puzzle.label = 'Erie'
erie_puzzle.description = '''
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
    P=dict(R=97,C=138,B=101),
    F=dict(S=99,B=211),
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

class loopy(search.Problem):
    def actions(self, state):
        return ['up', 'down', 'left', 'right']

    def result(self, state, action):
        if action == 'up':
            if state == 'A':
                return 'A'
            if state == 'B':
                return 'B'
            if state == 'C':
                return 'A'
            if state == 'D':
                return 'B'
        if action == 'down':
            if state == 'A':
                return 'C'
            if state == 'B':
                return 'D'
            if state == 'C':
                return 'C'
            if state == 'D':
                return 'D'
        if action == 'left':
            if state == 'A':
                return 'A'
            if state == 'B':
                return 'A'
            if state == 'C':
                return 'C'
            if state == 'D':
                return 'C'
        if action == 'right':
            if state == 'A':
                return 'B'
            if state == 'B':
                return 'B'
            if state == 'C':
                return 'D'
            if state == 'D':
                return 'D'

    def goal_test(self, state):
            return state == 'D'

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1



loopy_map = search.UndirectedGraph(dict(
    A=dict(B=1, D=1), B=dict(A=1, C=1, E=1), C=dict(B=1, F=1), D=dict(A=1, E=1)
))
loopy_map.locations = dict(
    A=(0,1), B=(1,1), C=(0,0), D=(1,0)
)

loopy_puzzle = loopy('A', loopy_map)




'''
loopy_map = search.UndirectedGraph(dict(
    A=dict(B=1, C=1), B=dict(A=1, D=1), C=dict(A=1, D=1), D=dict(B=1, C=1)
))


loopy_map = dict(
    A=(0, 1), B=(1, 1), C=(0, 0), D=(1, 0)
 )


loopy_puzzle = loopy('A', loopy_map)
'''
loopy_puzzle.label = 'Loopy'
loopy_puzzle.description = '''
An abbreviated map of the loopy puzzle
'''

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
#   swiss_puzzle,
    erie_puzzle,
    romania_puzzle,
    switch_puzzle,
    loopy_puzzle
]

mySearchMethods = []