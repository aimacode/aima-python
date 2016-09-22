import search
import math
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    Cottontown=dict(Portland=18),
    Fairfield=dict(Mitchellville=21, Portland=17),
    Mitchellville=dict(Portland=7, Fairfield=21),
))
#converts latitude to miles:
def latitude(lat):
    return lat * 69
#converts longitude to miles:
def longitude(lat,long):
    return long * 69 * math.cos(math.radians(lat))
kc_map = search.UndirectedGraph(dict(
    KansasCity=dict(Independence=11,OverlandPark=12,Atchison=50,),
    Independence=dict(Higginsville=46,LeesSummit=18,KansasCity=11),
    Higginsville=dict(Warrensburg=22,Independence=46),
    LeesSummit=dict(Warrensburg=39,Independence=18),
    Warrensburg=dict(Sedalia=30,Higginsville=22,LeesSummit=39),
    Sedalia=dict(Warsaw=35,Warrensburg=30),
    Warsaw=dict(Clinton=30, Sedalia=35),
    Clinton=dict(RichHill=51,Warsaw=30),
    RichHill=dict(Ottawa=87,Clinton=51),
    Ottawa=dict(OsageCity=32,RichHill=87),
    OsageCity=dict(Ottawa=32),
    OverlandPark=dict(Olathe=12,Lawrence=35,KansasCity=12,Holton=98),
    Olathe=dict(Lawrence=30,OverlandPark=12),
    Lawrence=dict(Topeka=28,OverlandPark=35,Olathe=30),
    Atchison=dict(KansasCity=50),
    Topeka=dict(Lawrence=28,StMarys=26),
    StMarys=dict(Topeka=26,Holton=37),
    Holton=dict(OverlandPark=98,StMarys=37),
))
kc_map.locations = dict(
    KansasCity=(latitude(39.0997), longitude(39.0997,94.5786)),
    Independence=(latitude(39.0911), longitude(39.0911,94.4155)),
    Higginsville=(latitude(39.0725), longitude(39.0725,93.7172)),
    LeesSummit=(latitude(38.9108), longitude(38.9108,94.3822)),
    Warrensburg=(latitude(38.7628), longitude(38.7628,93.7360)),
    Sedalia=(latitude(38.7045), longitude(38.7045,93.2283)),
    Warsaw=(latitude(38.2431), longitude(38.2431,93.3819)),
    Clinton= (latitude(38.3686), longitude(38.3686, 93.7783)),
    RichHill= (latitude(38.0964), longitude(38.0964, 94.3611)),
    Ottawa= (latitude(38.6158), longitude(38.6158, 95.2686)),
    OsageCity= (latitude(39.0000), longitude(38.6339, 95.8258)),
    OverlandPark= (latitude(38.9822), longitude(38.9822, 94.6708)),
    Olathe=(latitude(38.8814), longitude(38.8814,94.8191)),
    Lawrence=(latitude(38.9717), longitude(38.9717,95.2353)),
    Atchison=(latitude(39.5631), longitude(39.5631,95.1216)),
    Topeka=(latitude(39.0558), longitude(39.0558,95.6890)),
    StMarys=(latitude(39.1942), longitude(39.1942,96.0711)),
    Holton=(latitude(39.4653), longitude(39.4653,95.7364)),

)

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner Map'
sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''
kcmapTopeka_puzzle = search.GraphProblem('OsageCity','Topeka', kc_map)

kcmapTopeka_puzzle.label = 'Kansas City Map'
kcmapTopeka_puzzle.description = '''
A map of the Kansas City area in the Missouri-Kansas Bistate Area.
'''
kcmapStMarys_puzzle = search.GraphProblem('KansasCity','StMarys', kc_map)

kcmapStMarys_puzzle.label = 'Kansas City Map'
kcmapStMarys_puzzle.description = '''
A map of the Kansas City area in the Missouri-Kansas Bistate Area.
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

switch_puzzle = LightSwitch('off')
switch_puzzle.label = 'Light Switch'
#version of the sixteen puzzle that is 2 by 2
# class SixteenPuzzle(search.Problem):
#     def actions(self, state):
#         return ['uR', 'uL','dR','dL','lD','rD','rU','rD']
#
#     def result(self, state, action):
#         newState = state
#         if action == 'uR' or action == 'uL':
#             newState[2] = state[1]
#             newState[1] = state[2]
#         if action == 'dR' or action == 'dL':
#             newState[3] = state[4]
#             newState[4] = state[3]
#         if action == 'lD' or action == 'lU':
#             newState[1] = state[3]
#             newState[3] = state[1]
#         if action == 'rD' or action == 'rU':
#             newState[2] = state[4]
#             newState[4] = state[2]
#
#
#     def goal_test(self, state):
#         return state == ['1','2','3','4']
#
#     def h(self, node):
#         state = node.state
#         if self.goal_test(state):
#             return 0
#         else:
#             return 1
#
class SixteenPuzzle(search.Problem):
    def actions(self, state):
        return ['uR', 'uL','dR','dL','lD','rD','rU','rD']

    def result(self, state, action):
        newState = state
        a,b,c,d = newState
        if action == 'uR' or action == 'uL':
            a = b
            b = a
        if action == 'dR' or action == 'dL':
            c = d
            d = c
        if action == 'lD' or action == 'lU':
            a = c
            c = a
        if action == 'rD' or action == 'rU':
            b = d
            d = b
        newState = (a,b,c,d)
        return newState


    def goal_test(self, state):
        return state == ('1','2','3','4')

    def path_cost(self, c, state1, action, state2):
        return c+1
    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1
sixteen_puzzle = SixteenPuzzle(('2','3','4','1'))
myPuzzles = [
    kcmapTopeka_puzzle,
    kcmapStMarys_puzzle,
    sumner_puzzle,
    switch_puzzle,
    sixteen_puzzle,
]