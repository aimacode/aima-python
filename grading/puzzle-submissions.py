import agents as ag
# import envgui as gui
import importlib
import traceback
import search

class MyException(Exception):
    pass

roster = ['Ban','Becker','Blue','Capps','Conklin','Dickenson','Fritz',
          'Haller','Hawley','Hess','Johnson','Karman','Kinley','LaMartina',
          'McLean','Miles','Ottenlips','Porter','Sery','VanderKallen',
          'aardvark','zzzsolutions',
          ]

def compare_searchers(problems, header, searchers=[]):
    def do(searcher, problem):
        p = search.InstrumentedProblem(problem)
        goalNode = searcher(p)
        return p, goalNode.path_cost
    table = [[search.name(s)] + [do(s, p) for p in problems] for s in searchers]
    search.print_table(table, header)

submissions = {}
scores = {}

print('Submissions that compile: ')
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.puzzles')
        submissions[student] = mod.myPuzzles
        print('    ' + student)
    except:
        pass
print('----------------------------------------')

for student in submissions:
    scores[student] = []
    try:
        print(student)
        compare_searchers(
            problems=submissions[student],
            header=['',
                    '(<succ/goal/stat/fina>, cost)'
                    ],
            searchers=[
                search.depth_first_graph_search,
                search.breadth_first_search,
                search.iterative_deepening_search,
                search.uniform_cost_search,
            ]
        )
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')