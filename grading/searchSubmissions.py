import agents as ag
# import envgui as gui
import importlib
import traceback
import search

from utils import isnumber, memoize
from grading.util import roster, print_table
import inf

class MyException(Exception):
    pass

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


searches = {}
searchMethods = {}
scores = {}

messages = ['      Searches that compile:',
            'Search methods that compile:' ]
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.mySearches')
        searches[student] = mod.mySearches
        messages[0] += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()
    try:
        # http://stackoverflow.com/a/17136796/2619926
        #searchMethods[student] = submissions.bbbetter.mySearchMethods
        mod = importlib.import_module('submissions.' + student + '.mySearches')
        searchMethods[student] = mod.mySearchMethods
        messages[1] += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

for m in messages:
    print(m)
print('----------------------------------------')

def bestFS(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return search.best_first_graph_search(problem, lambda n: h(n))

for student in roster:
    if not student in searches.keys():
        continue
    scores[student] = []
    slist=[
        search.depth_first_graph_search,
        bestFS,
        search.breadth_first_search,
        search.iterative_deepening_search,
        search.uniform_cost_search,
        search.astar_search,
    ]
    if student in searchMethods:
        for s in searchMethods[student]:
            slist.append(s)
    try:
        plist = searches[student]
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
                searchers=slist
            )
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')
