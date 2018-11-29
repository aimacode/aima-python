import agents as ag
# import envgui as gui
import importlib
import traceback
import search
from math import inf

from utils import isnumber, memoize
from grading.util import roster, print_table
from inspect import signature
from math import inf

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
        try:
            goalNode = searcher(p)
            cost = goalNode.path_cost
            if cost < best[p.label]:
                best[p.label] = cost
                bestNode[p.label] = goalNode
            return p, cost
        except:
            # print('searcher(' + p.label + ') raised an exception:')
            # traceback.print_exc()
            return p, inf
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
                summary += " " + str(state)
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
        try:
            searches[student] = mod.mySearches
            messages[0] += ' ' + student
        except:
            print(student + ': mySearches[] is missing or defective.')
        try:
            searchMethods[student] = mod.mySearchMethods
            if len(searchMethods[student]) > 0:
                messages[1] += ' ' + student
        except:
            print(student + ': mySearchMethods[] is missing or defective.')
    except ImportError:
        # print('submissions/' + student + '/mySearches.py is missing or defective.')
        pass
    except:
        traceback.print_exc()

for m in messages:
    print(m)
print('----------------------------------------')

def bestFS(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return search.best_first_graph_search(problem, lambda n: h(n))

def wellFormed(problem):
    if not hasattr( problem, 'initial' ):
        print('problem "' + problem.label + '" has no initial state.')
        return False

    if not hasattr(problem, 'actions'):
        print('problem "' + problem.label + '" has no actions() method.')
        return False
    pasig = signature(problem.actions)
    if len(pasig.parameters) != 1:
        print('in problem "' + problem.label + '",')
        print('  actions(...) has the wrong number of parameters.  Define it as:')
        print('  def actions(self, state):')
        return False

    if not hasattr(problem, 'result'):
        print('problem "' + problem.label + '" has no result() method.')
        return False
    prsig = str(signature(problem.result))
    if len(pasig.parameters) != 2:
        print('in problem "' + problem.label + '",')
        print('  result(...) has the wrong number of parameters.  Define it as:')
        print('  def result(self, state, action):')
        return False

    if not hasattr(problem, 'goal_test'):
        if problem.goal == None:
            print('problem "' + problem.label + '" has no goal, and no goal_test() method.')
            return False
    pgsig = str(signature(problem.goal_test))
    if len(pgsig.parameters) != 1:
        print('in problem "' + problem.label + '",')
        print('  goal_test(...) has the wrong number of parameters.  Define it as:')
        print('  def goal_test(self, state):')
        return False
    return True

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
            if not wellFormed(problem):
                continue
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

    print(student, 'summary:', str(scores[student]), '\n' +
          student, '  total:', str(sum(scores[student])), '\n' +
          '----------------------------------------')
