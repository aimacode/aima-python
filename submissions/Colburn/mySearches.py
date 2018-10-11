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
    OldMission=(0,35),
    ElkRapids=(45,55),
    GlenArbor=(-65,35),
    Leland=(-45,45),
    Copmish =(-50,55),
    Interlochen=(-40,-30),
    BearLake=(-60,-60)
)





nomich_puzzle1= search.GraphProblem('TraverseCity', 'Copemish', nomich_map)
nomich_puzzle1.label='A puzzle where uniform-cost works best'
nomich_puzzle1.description='''
A puzzle where uniform-cost works best.
'''

nomich_puzzle2= search.GraphProblem('Interlochen', 'ElkRapids', nomich_map)
nomich_puzzle2.label='Breadth-First is better than Depth-First'
nomich_puzzle2.description='''
A puzzle where Breadth-First is better than Depth-First
'''

nomich_puzzle3= search.GraphProblem('Leland','ElkRapids' , nomich_map)
nomich_puzzle3.label='BFS better than bestFs'
nomich_puzzle3.description='''
A puzzle where Breadth-First is better than Depth-First
'''

nomich_puzzle4= search.GraphProblem('Interlochen','Leland' , nomich_map)
nomich_puzzle4.label='A* expands less nodes with same cost as UFC'
nomich_puzzle4.description='''
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


def String2State(myString):
    a0 = "A1,A2,A3,B1,B2,B3,C1,C2,C3".split(",")
    a1 = myString.split(",")
    state = [[],[],[],
             [],[],[],
             [],[],[]]
    for i in range(len(state)):
        state[i] = [a0[i], int(a1[i])]
    return state
def State2String(state):
    string =""
    for i in range(len(state)):
        if i < len(state)-1:
            string = string + str(state[i][1])+","
        else:
            string = string + str(state[i][1])
    return string.strip()
def getBest(state,goal):
    string_state=state.state.split(",")
    goals = goal.split(",")
    goal_state=[]
    new_state=[]
    for j in range(len(string_state)):
        new_state.append(int(string_state[j]))
        goal_state.append(int(goals[j]))

    #print(goal_state)
    #print(new_state)

    best=0
    for i in range(len(new_state)):
        #print(new_state)
        best = (abs(i-goal_state.index(new_state[i]))) + best
    return best






class SlidingPuzzle(search.Problem):

    def __init__(self,initial,goal):
        self.initial = initial
        self.goal = goal

    def actions(self,state):
        state1 = String2State(state)[:]
        #print(state1)

        x = "blank"
        for i in range(len(state1)):
            #print(state1[i][1])
            if state1[i][1] == 0:
                x = state1[i][0]
                break
            #print(x)

        #print('space:'+ str(x))
        if x == 'A1':
            return[1,3]
        elif x == 'A2':
            return[0,2,4]
        elif x == 'A3':
            return[1,5]
        elif x =='B1':
            return[0,4,6]
        elif x =='B2':
            return[3,1,5,7]
        elif x =='B3':
            #print([4,2,8])
            return[4,2,8]
        elif x =='C1':
            return[7,3]
        elif x =='C2':
            return[8,4,6]
        elif x =='C3':
            return[7,5]


    def result(self,state, action):
        state1 = String2State(state)[:]
        #print('resultIn: '+str(state1))
        for i in range(len(state1)):
            #print(state1[i][1])
            if state1[i][1] == 0:
                x = i
                break
        #print(x)
        #print(action)

        state1[x][1] = state1[action][1]
        state1[action][1] = 0
        #print('new state: '+str(state1))
        #print(State2String(state1))
        return State2String(state1)

    def goal_test(self,state):
        #print('state: ' + state)
        #print('goal: '+ self.goal)
        return state == self.goal
    def h(self,state):
        return getBest(state,self.goal)

















#swiss_puzzle = search.GraphProblem('A', 'Z', sumner_map)

start_puzzle1      = "1,2,3,4,5,0,7,8,6"
SlidingPuzzle_Goal = "1,2,3,4,5,6,7,8,0"
start_puzzle2      = "1,2,3,4,0,6,5,7,8"

slide_puzzle1 = SlidingPuzzle(str(start_puzzle1), str(SlidingPuzzle_Goal))
slide_puzzle1.label ='SlidepuzzleTest'

slide_puzzle2= SlidingPuzzle(start_puzzle2,SlidingPuzzle_Goal)
slide_puzzle2.label= "Slidepuzzle2"

switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'

mySearches = [
 #   swiss_puzzle,
 #   sumner_puzzle,
 #  romania_puzzle,
    #switch_puzzle,
    nomich_puzzle1,
    nomich_puzzle2,
    nomich_puzzle3,
    nomich_puzzle4,
    slide_puzzle1,
    #slide_puzzle2
]
mySearchMethods = []
