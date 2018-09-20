import search
from math import(cos, pi)

# A sample map problem
# sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
# ))
#
# sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)
#
# sumner_puzzle.label = 'Sumner'
# sumner_puzzle.description = '''
# An abbreviated map of Sumner County, TN.
# This map is unique, to the best of my knowledge.
# '''

#=========================================================================
#=========================================================================
norfolk_map = search.UndirectedGraph(dict(
    Norfolk=dict(Suffolk=50,Chesapeake=15,VirginiaBeach=35),
    Suffolk=dict(Norfolk=50,Chesapeake=35,Hampton=60,Moyock=150,Sunbury=120),
    Chesapeake=dict(Suffolk=35,Norfolk=15,VirginiaBeach=40,Moyock=120),
    VirginiaBeach=dict(Norfolk=35,Chesapeake=40),
    Hampton=dict(Norfolk=30,Suffolk=60,NewportNews=15),
    NewportNews=dict(Hampton=15,Jamestown=35,Williamsburg=30,Yorktown=15),
    Jamestown=dict(NewportNews=35,Williamsburg=15),
    Williamsburg=dict(Jamestown=15,NewportNews=30,Yorktown=20),
    Yorktown=dict(Williamsburg=20,Newportnews=15),
    Sunbury=dict(Suffolk=120, Moyock=45),
    Moyock=dict(Suffolk=150,Chesapeak=120),
))

norfolk_puzzle = search.GraphProblem('Jamestown', 'Yorktown', norfolk_map)

norfolk_puzzle.label = 'Norfolk'
norfolk_puzzle.description = 'This is a map of the Norfolk, VA area.' \
                             'This map is unique to the best of my' \
                             'knowledge.'

#=========================================================================
#=========================================================================


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
 #   sumner_puzzle,
    romania_puzzle,
    switch_puzzle,
    norfolk_puzzle,
]

