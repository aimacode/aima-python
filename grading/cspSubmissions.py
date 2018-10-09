import importlib
import traceback
from grading.util import roster, print_table
from csp import lcv, mrv, mac, first_unassigned_variable, unordered_domain_values, no_inference
# from search import InstrumentedProblem

def backtracking_search(csp,
                        select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values,
                        inference=no_inference):
    """[Figure 6.5]
    """

    def backtrack(assignment):
        csp.bcount += 1
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            csp.acount += 1
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    csp.bcount = 0
    csp.acount = 0
    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result

def try_csps(csps):
    i = 1
    for c in csps:
        problem = c['csp']
        try:
            label = problem.label
        except:
            label = 'Problem'
        if len(c) > 1:
            def getfunctionname(f):
                return f.__name__
            optionlist = list(c.values())[1:]
            optionnames = list(map(getfunctionname, optionlist))
            label += ' ' + str(optionnames)
        label += ' (' + str(i) + '): '
        i += 1
        # insp = InstrumentedProblem(problem)
        # p = { 'csp' : insp }
        # for k in c.keys():
        #     if k == 'csp':
        #         continue
        #     p[k] = c[k]
        assignment = backtracking_search(
            **c
            # **p
            # order_domain_values=lcv,
            # select_unassigned_variable=mrv,
            # inference=mac
        )

        print(label + str(problem.bcount) + ' recursive calls, '
              + str(problem.acount) + ' assignments tested.')
        print(assignment)
        print()

submissions = {}
scores = {}

message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myCSPs')
        submissions[student] = mod.myCSPs
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

print(message1)
print('----------------------------------------')

for student in roster:
    if not student in submissions.keys():
        continue
    scores[student] = []
    try:
        csps = submissions[student]
        print('CSPs from:', student)
        try_csps(csps)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')
