import agents as ag
# import envgui as gui
import importlib
import traceback
import search
from utils import(isnumber)

class MyException(Exception):
    pass

roster = ['Ban','Becker','Blue','Capps','Conklin','Dickenson','Fritz',
          'Haller','Hawley','Hess','Johnson','Karman','Kinley','LaMartina',
          'McLean','Miles','Ottenlips','Porter','Sery','VanderKallen',
          'aardvark','zzzsolutions',
          ]


def print_table(table, header=None, sep='   ', numfmt='%g'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        r = 0
        for row in header:
            table.insert(r, row)
            r += 1

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))


def compare_searchers(problems, header, searchers=[]):
    def do(searcher, problem):
        p = search.InstrumentedProblem(problem)
        goalNode = searcher(p)
        return p, goalNode.path_cost
    table = [[search.name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

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
        plist = submissions[student]
        hlist = [[student],['']]
        i = 0
        for problem in plist:
            try:
                hlist[0].append(problem.label)
            except:
                hlist[0].append('Problem ' + str(i))
            i += 1
            hlist[1].append('(<succ/goal/stat/fina>, cost)')
        compare_searchers(
            problems=plist,
            header=hlist,
            searchers=[
                search.depth_first_graph_search,
                search.breadth_first_search,
                search.iterative_deepening_search,
                search.uniform_cost_search,
                search.astar_search,
            ]
        )
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')