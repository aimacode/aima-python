import search
from math import(cos, pi)

# A sample map problem
sumner_map = search.UndirectedGraph(dict(
    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    Cottontown=dict(Portland=18),
    Fairfield=dict(Mitchellville=21, Portland=17),
    Mitchellville=dict(Portland=7, Fairfield=21),
))
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
    OsageCity=dict(Topeka=38,Ottawa=32),
    OverlandPark=dict(Olathe=12,Lawrence=35,KansasCity=12,Holton=98),
    Olathe=dict(Lawrence=30,OverlandPark=12),
    Lawrence=dict(Topeka=28,OverlandPark=35,Olathe=30),
    Atchison=dict(KansasCity=50),
    Topeka=dict(Lawrence=28,OsageCity=38,StMarys=26),
    StMarys=dict(Topeka=26,Holton=37),
    Holton=dict(OverlandPark=98,StMarys=37),
))

sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

sumner_puzzle.label = 'Sumner Map'
sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''
kcmapTopeka_puzzle = search.GraphProblem('KansasCity','Topeka', kc_map)

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

myPuzzles = [
    #kcmapTopeka_puzzle,
    kcmapStMarys_puzzle,
    sumner_puzzle,
    switch_puzzle,
]