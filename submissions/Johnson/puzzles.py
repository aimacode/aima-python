import search
from math import(cos, pi)
sumner_map = search.UndirectedGraph(dict(
    #Portland=dict(Mitchellville=7, Fairfield=17, Cottontown=18),
    #Cottontown=dict(Portland=18),
    #Fairfield=dict(Mitchellville=21, Portland=17),
    #Mitchellville=dict(Portland=7, Fairfield=21),

#cost is in estimated minutes of drive time from
#https://distancefrom.co.uk
#https://distancefrom.co.uk/from-machynlleth-to-dolgellau for example
    Newtown=dict(Machynlleth=46, Dolgellau=61, Conwy=113, Bangor=131, Caernarnfon=123, Betws_y_coed=110,
                 Pwllheli=117, Llangollen=63, Welshpool=22, Aberystwyth=70),
    Machynlleth=dict(Newtown=46, Dolgellau=27, Conwy=100, Bangor=103, Caernarnfon=88, Betws_y_coed=74, Wrexham= 93,
                      Llangollen = 81, Welshpool= 57, Aberystwyth= 33),
    Dolgellau=dict(Newtown=61, Machynlleth=27, Conwy=77, Bangor=81, Caernarnfon=65, Betws_y_coed=52, Wrexham=78,
                    Llangollen=63, Welshpool=57, Aberystwyth=60),
    Conwy=dict(Newtown= 113, Machynlleth= 100, Dolgellau= 77, Bangor=24, Caernarnfon=31, Betws_y_coed=31, Wrexham=60,
                Llangollen=72, Welshpool=96, Aberystwyth=133),
    Bangor=dict(Newtown= 131, Machynlleth= 103, Dolgellau= 81, Conwy=24, Caernarnfon=18, Betws_y_coed=37, Wrexham=77,
                    Llangollen=86, Welshpool=113, Aberystwyth=136),
    Caernarnfon=dict(Newtown= 123, Machynlleth= 88, Dolgellau= 65, Conwy=31, Bangor=18, Betws_y_coed=44, Wrexham=86,
                   Pwllheli=34, Llangollen=93, Welshpool=117, Aberystwyth=121),
    Betws_y_coed=dict(Newtown= 110, Machynlleth= 74, Dolgellau= 52, Conwy=31, Bangor=37, Caernarnfon=44, Wrexham=67,
                   Pwllheli=61, Llangollen=51, Welshpool=89, Aberystwyth=108),
    Wrexham=dict(Machynlleth= 93, Dolgellau= 78, Conwy=60, Bangor=77, Caernarnfon=86, Betws_y_coed=67,
                   Pwllheli=113, Llangollen=22, Aberystwyth=126),
    Pwllheli=dict(Newtown= 117, Caernarnfon=34, Betws_y_coed=61,
                   Wrexham=113, Llangollen=96, Welshpool=111, Aberystwyth=114),
    Llangollen=dict(Newtown= 63, Machynlleth= 81, Dolgellau= 63, Conwy=72, Bangor=86, Caernarnfon=93, Betws_y_coed=51,
                   Wrexham=22, Pwllheli=96, Welshpool=45, Aberystwyth=114),
    Welshpool=dict(Newtown= 22, Machynlleth= 57, Dolgellau= 57, Conwy=96, Bangor=113, Caernarnfon=117, Betws_y_coed=89,
                Pwllheli=111, Llangollen=45, Aberystwyth=90),
    Aberystwyth=dict(Newtown= 70, Machynlleth= 33, Dolgellau= 60, Conwy=133, Bangor=136, Caernarnfon=121, Betws_y_coed=108,
                   Wrexham=126, Pwllheli=114, Llangollen=114, Welshpool=90)))
sumner_map.locations = dict(Newtown=(525121,33131), Machynlleth=(525903,38535),
                                Dolgellau=(527421,38844), Conwy=(532829,38295),
                                Bangor=(532274,41293), Caernarnfon=(531396,42739),
                                Betws_y_coed=(530931, 38010), Wrexham=(530430, 29925),
                                Pwllheli=(528888,44176), Llangollen=(529692,31717),
                                Welshpool=(526603,31464), Aberystwyth=(524153,40829))

#all instances run BestFS and A*
#sumner_puzzle yields better solution for BestFS than DFS, and BFS better than BestFS
sumner_puzzle = search.GraphProblem('Pwllheli', 'Conwy', sumner_map)

#sumner1_puzzle adds nothing new
sumner1_puzzle = search.GraphProblem('Pwllheli', 'Newtown', sumner_map)

#sumner2_puzzle yields same solution with UCS and A*, but A* expands fewer nodes
sumner2_puzzle = search.GraphProblem('Newtown', 'Wrexham', sumner_map)

sumner_puzzle.description = '''
An abbreviated map of Sumner County, TN.
This map is unique, to the best of my knowledge.
'''

#cannot for the life of me remember how to get the table to print every problem instance.
myPuzzles = [
    sumner_puzzle,
    sumner1_puzzle,
    sumner2_puzzle
]
