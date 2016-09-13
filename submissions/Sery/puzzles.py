import search
from math import(cos, pi)

mediterranean_map = search.UndirectedGraph(dict(
    Alexandria=dict(Rome=53, Byzantium=18, Crete=12, Cyprus=6.5, Cyrene=8, Massilia=58, Myra=6, Naples=60, Rhodes=8),
    Ascalon=dict(Thessalonica=15),
    Berytus=dict(Rhodes=8),
    Byzantium=dict(Alexandria=9, Gaza=5, Rhodes=10),
    Caesarea=dict(Rhodes=10),
    Carthage=dict(Gibraltar=7, Rome=3),
    Corinth=dict(Naples=5),
    Crete=dict(Alexandria=3.5, Cyrene=1.5),
    Cyprus=dict(Alexandria=2, Rhodes=4.5),
    Cyrene=dict(Alexandria=4.5, Crete=2),
    Epidamnus=dict(Rome=5),
    Gaza=dict(Byzantium=5, Rhodes=11),
    Gibraltar=dict(Rome=8, Carthage=7),
    Massilia=dict(Rome=2.5, Alexandria=25),
    Myra=dict(Alexandria=3),
    Naples=dict(Corinth=7, Alexandria=10, Rome=3),
    Narbo=dict(Utica=5, Rome=3),
    Rhodes=dict(Alexandria=3.5, Berytus=3.5, Byzantium=10, Caesarea=3.5, Cyprus=2, Gaza=3.5, Rome=52, Tyre=4),
    Rome=dict(Alexandria=11, Carthage=3, Epidamnus=5, Gibraltar=7, Massilia=5, Naples=1, Narbo=3, Rhodes=9, Tarraco=4),
    Tarraco=dict(Rome=4),
    Thessalonica=dict(Ascalon=12),
    Tyre=dict(Rhodes=10),
))

mediterranean_puzzle = search.GraphProblem('Gibraltar', 'Alexandria', mediterranean_map)

mediterranean_puzzle.description = '''
An abbreviated map of ports in the Mediterranean Sea.
Times are based from the article "Speed Under Sail of Ancient Ships", a journal released by the University of Chicago.
http://penelope.uchicago.edu/Thayer/E/Journals/TAPA/82/Speed_under_Sail_of_Ancient_Ships*.html
'''

myPuzzles = [
    mediterranean_puzzle,
    search.GraphProblem('Alexandria', 'Naples', mediterranean_map),
    search.GraphProblem('Tyre', 'Gaza', mediterranean_map),
    search.GraphProblem('Narbo', 'Corinth', mediterranean_map),
]