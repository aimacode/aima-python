import search

from math import(cos, pi)
import numpy
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

nomich_map.location = dict(
   TraverseCity=(0,0),
    OldMission=(0,15),
    ElkRapids=(25,15),
    GlenArbor=(-45,15),
    Leland=(-25,25),
    Copmish =(-30,35),
    Interlochen=(-20,-10),
    BearLake=(-40,-40)
)





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






class SlidingPuzzle(search.Problem):

    def __init__(self,initial,goal,board):
        self.initial = initial
        self.goal = goal
        self.boardMap = board
        self.currentstate= initial
        self.cost=0
        print(self.initial)

    def actions(self,state):

        for i in state:
            if i[1] == 0:
                space = i[0]
        #print(self.boardMap.get(space).keys())


        return [self.boardMap.get(space).values()]

    def result(self,state, action):
        new_state = state
        count = 0
        for i in new_state:

            if i[1] == 0:
                dex = count
                break
            count = count + 1
        print(dex)
        print(new_state[action])
        new_state[dex][1] = new_state[action][1]
        new_state[action][1] = 0
        return new_state

    def goal_test(self,state):
        return self.currentstate == self.goal













#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)

SlidingPuzzle_Board=dict(
     A1=dict(right=1,down=3), A2 =dict(left=0,right=2,down=4), A3=dict(left=1,down=5)

    ,B1=dict(up=0,right=4,down=6), B2=dict(left=3,up=1,right=5,down=7), B3=dict(left=4,up=2,down=8)

    ,C1=dict(right=7,up = 3), C2=dict(right=8,up=4,left=6), C3=dict(left=7,up=5)
    )


SlidingPuzzle_Goal = [['A1', 1], ['A2', 2], ['A3', 3],
                      ['B1', 4], ['B2', 5], ['B3', 6],
                     ['C1', 7], ['C2', 8], ['C3', 0]]

#SlidingPuzzle_Goal = [1,2,3,4,5,6,7,8,0]



#SlidingPuzzle_Goal=dict(
#     A1=dict(value=1, right='A2',down='B1'), A2 =dict(left='A1',right=,down=4), A3=dict(left=1,down=5)

#    ,B1=dict(up=0,right=4,down=6), B2=dict(left=3,up=1,right=5,down=7), B3=dict(left=4,up=2,down=8)

#    ,C1=dict(right=7,up = 3), C2=dict(right=8,up=4,left=6), C3=dict(left=7,up=5)
#    )

start_puzzle1 = [['A1', 1], ['A2', 2], ['A3', 3],
                 ['B1', 4], ['B2', 5], ['B3', 0],
                 ['C1', 7], ['C2', 8], ['C3', 6]]

slide_puzzle1 = SlidingPuzzle(start_puzzle1, SlidingPuzzle_Goal, SlidingPuzzle_Board)
slide_puzzle1.label ='Slidepuzzle'

switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
 #   sumner_puzzle,
 #  romania_puzzle,
    #switch_puzzle,
    nomich_puzzle1,
    nomich_puzzle2,
    slide_puzzle1
]
mySearchMethods = []
