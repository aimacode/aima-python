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
fayette_puzzle.label = 'Fayette County'
fayette_puzzle.description = '''
An abbreviated map of Fayette County, PA.
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

mySearches = [
 #   swiss_puzzle,
   # sumner_puzzle,
    romania_puzzle,
    switch_puzzle,
    fayette_puzzle,
]

mySearchMethods = []