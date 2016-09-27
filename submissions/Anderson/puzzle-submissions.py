import agents as ag
# import envgui as gui
import importlib
import traceback
import search
from utils import(isnumber, memoize)
from math import(inf)

class MyException(Exception):
    pass

roster = ['Anderson', 'Ban','Becker','Blue','Capps','Conklin','Dickenson',
          'Fritz','Haller','Hawley','Hess','Johnson','Karman','Kinley',
          'LaMartina','McLean','Miles','Ottenlips','Porter','Sery',
          'VanderKallen',
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
    best = {}
    bestNode = {}
    for p in problems:
        best[p.label] = inf
        bestNode[p.label] = None
    def do(searcher, problem):
        nonlocal best, bestNode
        p = search.InstrumentedProblem(problem)
        goalNode = searcher(p)
        cost = goalNode.path_cost
        if cost < best[p.label]:
            best[p.label] = cost
            bestNode[p.label] = goalNode
        return p, cost
    table = [[search.name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)
    print('----------------------------------------')
    for p in problems:
        bestPath = []
        node = bestNode[p.label]
        while node != None:
            bestPath.append(node.state)
            node = node.parent
        summary = "Best Path for " + p.label + ": "
        for state in reversed(bestPath):
            try:
                summary += "\n" + p.prettyPrint(state) + "\n---------"
            except:
                summary += " " + state
        print(summary)
        print('----------------------------------------')


submissions = {}
scores = {}

message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.puzzles')
        submissions[student] = mod.myPuzzles
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

print(message1)
print('----------------------------------------')

def bestFS(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return search.best_first_graph_search(problem, lambda n: h(n))

for student in roster:
    if not student in submissions.keys():
        continue
    scores[student] = []
    try:
        plist = submissions[student]
        hlist = [[student],['']]
        i = 0
        for problem in plist:
            try:
                hlist[0].append(problem.label)
            except:
                problem.label = 'Problem ' + str(i)
                hlist[0].append(problem.label)
            i += 1
            hlist[1].append('(<succ/goal/stat/fina>, cost)')
        compare_searchers(
            problems=plist,
            header=hlist,
            searchers=[
                search.depth_first_graph_search,
                bestFS,
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