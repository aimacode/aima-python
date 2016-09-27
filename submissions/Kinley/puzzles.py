import search
from math import(cos, pi)

swiss_map = search.UndirectedGraph(dict(
    Basel=dict(Bern=63, Biel=59, Fribourg=81,
               Lausanne=127, Lucerne=62, Lugano=166, Schaffhausen=85,
               Sion=157, Thun=77, Winterhur=67, Zug=69, Zurich=54),
    Bern=dict(Basel=63, Biel=27, Fribourg=20,
              Lausanne=64, Lucerne=70, Lugano=173, Schaffhausen=108,
              Sion=96, Thun=18, Winterhur=90, Zug=84),
    Biel=dict(Basel=59, Fribourg=44,
              Lausanne=66, Lugano=169, Schaffhausen=104,
              Sion=120,),
    Chur=dict(
             Lausanne=190, Lucerne=92, Lugano=111,
              Zug=74, Zurich=81,),
    Fribourg=dict(Basel=81, Bern=20, Biel=44, Chur=159,
              Geneva=79, Lausanne=39, Lucerne=72, Lugano=177, Schaffhausen=111,
              Sion=81, Thun=31, Winterhur=109, Zug=89, Zurich=94,),
    Geneva=dict(Fribourg=79,
                Lausanne=37, Lucerne=145, Lugano=211, Schaffhausen=181,
              Sion=85, Winterhur=179,),
    Lausanne=dict(Basel=127, Bern=64, Biel=66, Chur=190, Fribourg=39,
              Geneva=37, Lucerne=108, Lugano=50, Schaffhausen=144,
              Sion=60, Thun=69, Winterhur=142, Zug=126, Zurich=129,),
    Lucerne=dict(Basel=62, Bern=70, Chur=92, Fribourg=72,
              Geneva=145, Lausanne=108, Lugano=146, Schaffhausen=68,
              Sion=110, Thun=49, Zug=16,),
    Lugano=dict(Basel=166, Bern=173, Biel=169, Chur=111, Fribourg=177,
              Geneva=211, Lausanne=50, Lucerne=146, Schaffhausen=202,
              Sion=125, Thun=145, Zug=160, Zurich=177,),
    Schaffhausen=dict(Basel=85, Bern=108, Biel=104, Fribourg=111,
              Geneva=181, Lausanne=144, Lucerne=68, Lugano=202,
              Sion=163, Thun=103, Winterhur=18, Zurich=31,),
    Sion=dict(Zurich=545, Basel=157, Bern=96, Biel=120, Fribourg=81,
              Geneva=85, Lausanne=60, Lucerne=110, Lugano=125, Schaffhausen=163,
             Winterhur=157, ),
    Thun=dict(Basel=77, Bern=18, Fribourg=31,
              Lausanne=69, Lucerne=49, Lugano=145, Schaffhausen=103,
              Winterhur=97, Zurich=83,),
    Winterhur=dict(Basel=67, Bern=90, Fribourg=109,
              Geneva=179, Lausanne=142, Schaffhausen=18,
              Sion=157, Thun=97, Zug=35, Zurich=15,),
    Zug=dict(Basel=69, Bern=84, Chur=74, Fribourg=89,
              Lausanne=126, Lucerne=16, Lugano=160,
             Winterhur=35, Zurich=20,),
    Zurich=dict(Basel=54, Bern=77, Chur=81, Fribourg=94,
              Lausanne=129, Lugano=177, Schaffhausen=31,
              Sion=545, Thun=83, Winterhur=15, Zug=20)
))
'''
swiss_map.locations = dict(
    Basel=(476,76), Bern=(469,74), Biel=(471,72), Chur=(469,95),Fribourgh=(480,78), Geneva=(462,61),
    Lausanne=(465,66), Lugano=(460,90), Schaffhausen=(477,86), Sion=(462,74), Thun=(468,76),
    Winterhur=(475,87), Zug=(472,65), Zurich=(474,85))
'''

swiss_puzzle = search.GraphProblem('Chur', 'Biel', swiss_map)
swiss_puzzle1 = search.GraphProblem('Zug', 'Geneva', swiss_map)
swiss_puzzle2 = search.GraphProblem('Chur', 'Schaffhausen', swiss_map)
swiss_puzzle3 = search.GraphProblem('Zurich', 'Zug', swiss_map)

swiss_puzzle.description = '''
An abbreviated map of major cities in Switzerland.
'''

# A trivial Problem definition
#class Hex(search.Problem):
   # def actions(self, state):
   #     return ['nw', 'ne', 'e', 'se','sw','w']
    #     All of the grid
    #     ['(0,0)', '(0,1)', '(0,2)'],
    #     ['(1,0)', '(1,1)', '(1,2)', '(1,3)'],
    #     ['(2,0)', '(2,1)', '(2,2)', '(2,3)', '(2,4)'],
    #     ['(3,0)', '(3,1)', '(3,2)', '(3,3'],
    #     ['(4,0', '(4,1)', '(4,2)']
    #     ])

#    hex_map = search.UndirectedGraph(dict(00=dict=(01=1,

#    puzzle= new Hex([['(0,0)', '(0,1)'],
#     ['(1,0)', '(1,1)', '(1,3)'],
#     ['(2,0)', '(2,1)', '(2,2)', '(2,3)', '(2,4)'],
#     ['(3,0)', '(3,2)', '(3,3'],
#     ['(4,0', '(4,1)', '(4,2)']
#     ])

 #   def result(self, state, action):
       # if action == 'up':
       #     return 'on'
       # else:
       #     return 'off'
   #     if action == 'nw':
   #         return 'go'
   #     elif action == 'ne':
   #         return 'go'
   #     elif action == 'e':
   #         return 'go'
   #     elif action == 'se':
   #         return 'go'
   #     elif action == 'sw':
   #         return 'go'
   #     elif action == 'w':
   #         return 'go'


myPuzzles = [
    swiss_puzzle1,
    swiss_puzzle2,
    swiss_puzzle3,
]