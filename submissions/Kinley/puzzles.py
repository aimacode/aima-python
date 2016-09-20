import search
from math import(cos, pi)

swiss_map = search.UndirectedGraph(dict(
    Basel=dict(Bern=63, Biel=59, Chur=128, Fribourg=81, Geneva=158,
               Lausanne=127, Lucerne=62, Lugano=166, Schaffhausen=85,
               Sion=157, Thun=77, Winterhur=67, Zug=69, Zurich=54),
    Bern=dict(Basel=63, Biel=27, Chur=151, Fribourg=20,
              Geneva=99, Lausanne=64, Lucerne=70, Lugano=173, Schaffhausen=108,
              Sion=96, Thun=18, Winterhur=90, Zug=84
             #, Zurich=77,
              ),
    Biel=dict(Basel=59,
              #Bern=27,
              Chur=147, Fribourg=44,
              Geneva=97, Lausanne=66, Lucerne=66, Lugano=169, Schaffhausen=104,
              Sion=120, Thun=40, Winterhur=88, Zug=76, Zurich=74,),
    Chur=dict(Basel=128, Bern=151, Biel=147, Fribourg=159,
              Geneva=217, Lausanne=190, Lucerne=92, Lugano=111, Schaffhausen=93,
              Sion=133, Thun=129, Winterhur=81, Zug=74, Zurich=81,),
    Fribourg=dict(Basel=81, Bern=20, Biel=44, Chur=159,
              Geneva=79, Lausanne=39, Lucerne=72, Lugano=177, Schaffhausen=111,
              Sion=81, Thun=31, Winterhur=109, Zug=89, Zurich=94,),
    Geneva=dict(Basel=158, Bern=99, Biel=97, Chur=217, Fribourg=79,
                Lausanne=37, Lucerne=145, Lugano=211, Schaffhausen=181,
              Sion=85, Thun=106, Winterhur=179, Zug=163, Zurich=166,),
    Lausanne=dict(Basel=127, Bern=64, Biel=66, Chur=190, Fribourg=39,
              Geneva=37, Lucerne=108, Lugano=185, Schaffhausen=144,
              Sion=60, Thun=69, Winterhur=142, Zug=126, Zurich=129,),
    Lucerne=dict(Basel=62, Bern=70, Biel=66, Chur=92, Fribourg=72,
              Geneva=145, Lausanne=108, Lugano=146, Schaffhausen=68,
              Sion=110, Thun=49, Winterhur=47, Zug=16, Zurich=33,),
    Lugano=dict(Basel=166, Bern=173, Biel=169, Chur=111, Fribourg=177,
              Geneva=211, Lausanne=185, Lucerne=146, Schaffhausen=202,
              Sion=125, Thun=145, Winterhur=191, Zug=160, Zurich=177,),
    Schaffhausen=dict(Basel=85, Bern=108, Biel=104, Chur=93, Fribourg=111,
              Geneva=181, Lausanne=144, Lucerne=68, Lugano=202,
              Sion=163, Thun=103, Winterhur=18, Zug=50, Zurich=31,),
    Sion=dict(Basel=157, Bern=96, Biel=120, Chur=133, Fribourg=81,
              Geneva=85, Lausanne=60, Lucerne=110, Lugano=125, Schaffhausen=163,
               Thun=61, Winterhur=157, Zug=128, Zurich=145,),
    Thun=dict(Basel=77, Bern=18, Biel=40, Chur=129, Fribourg=31,
              Geneva=106, Lausanne=69, Lucerne=49, Lugano=145, Schaffhausen=103,
              Sion=61, Winterhur=97, Zug=67, Zurich=83,),
    Winterhur=dict(Basel=67, Bern=90, Biel=88, Chur=81, Fribourg=109,
              Geneva=179, Lausanne=142, Lucerne=47, Lugano=191, Schaffhausen=18,
              Sion=157, Thun=97, Zug=35, Zurich=15,),
    Zug=dict(Basel=69, Bern=84, Biel=76, Chur=74, Fribourg=89,
              Geneva=163, Lausanne=126, Lucerne=16, Lugano=160, Schaffhausen=50,
              Sion=128, Thun=267, Winterhur=35, Zurich=20,),
    Zurich=dict(Basel=54, Bern=77, Biel=74, Chur=81, Fribourg=94,
              Geneva=166, Lausanne=129, Lucerne=33, Lugano=177, Schaffhausen=31,
              Sion=145, Thun=83, Winterhur=15, Zug=20)
))

swiss_puzzle = search.GraphProblem('Bern', 'Fribourg', swiss_map)
#swiss_puzzle1 = search.GraphProblem('Bern', 'Fribourg', swiss_map)
swiss_puzzle2 = search.GraphProblem('Zug', 'Thun', swiss_map)

swiss_puzzle.description = '''
An abbreviated map of major cities in Switzerland.
'''

myPuzzles = [
    swiss_puzzle,
    swiss_puzzle2
#    search.GraphProblem('Bern', 'Zurich', swiss_map),
#    search.GraphProblem('Lucerne', 'Basel', swiss_map),
#    search.GraphProblem('Winterhur', 'Sion', swiss_map),
]


#sumner_map = search.UndirectedGraph(dict(
#    Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
#    Cottontown=dict(Portland=18),
#    Fairfield=dict(Mitchellville=21, Portland=17),
#    Mitchellville=dict(Portland=7, Fairfield=21),
#))

#sumner_puzzle = search.GraphProblem('Cottontown', 'Mitchellville', sumner_map)

#sumner_puzzle.description = '''
#An abbreviated map of Sumner County, TN.
#This map is unique, to the best of my knowledge.
#'''

#myPuzzles = [
#    sumner_puzzle,
#]