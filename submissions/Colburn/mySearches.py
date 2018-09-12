import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
   Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
   Cottontown=dict(Portland=18),
   Fairfield=dict(Mitchellville=21, Portland=17),
   Mitchellville=dict(Portland=7, Fairfield=21),
))
'''
sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner'
sumner_puzzle.description = 
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
'''
romania_puzzle = search.GraphProblem('A', 'B', romania_map)

romania_puzzle.label = 'Romania'
romania_puzzle.description = 
The simplified map of Romania, per
Russall & Norvig, 3rd Ed., p. 68.
'''
nomich_map = search.UndirectedGraph(dict(
    TraverseCity = dict(Interlochen=20, Leland=38, GlenArbor=37, OldMission=27,Copemish=43,ElkRapids=30),
    Interlochen = dict(TraverseCity=23, Copemish=20, Leland=42,BearLake=62),
    Leland=dict(TraverseCity=38, GlenArbor=24, Copemish=62, OldMission=61),
    GlenArbor=dict(TraverseCity=37, ElkRapids=76),
    OldMission=dict(TraverseCity=27),
    Copmish = dict(TraverseCity=43, Interlochen=23, BearLake=21),
    ElkRapids= dict(TraverseCity=30, GlenArbor=76)
))
#
nomich_puzzle1= search.GraphProblem('TraverseCity', 'Copemish', nomich_map)
nomich_puzzle1.label='TraverseCity to Copemish'
nomich_puzzle1.description='''
A puzzle where uniform-cost works best.
'''

nomich_puzzle2= search.GraphProblem('Interlochen', 'ElkRapids', nomich_map)
nomich_puzzle2.label='Interlochen to Elkrapids'
nomich_puzzle2.description='''
A puzzle where Breadth-First is better than Depth-First
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
 #   sumner_puzzle,
 #  romania_puzzle,
    #switch_puzzle,
    nomich_puzzle1,
    nomich_puzzle2
]
mySearchMethods= []
