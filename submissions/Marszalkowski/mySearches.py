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

#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)
switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
    erie_puzzle,
    romania_puzzle,
    switch_puzzle,
]

mySearchMethods = []