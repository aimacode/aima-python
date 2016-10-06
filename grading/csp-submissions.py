import importlib
import traceback
from grading.util import roster, print_table
from csp import backtracking_search, lcv, mrv, mac

def try_csps(csps):
    for c in csps:
        assignment = backtracking_search(
            **c
            # order_domain_values=lcv,
            # select_unassigned_variable=mrv,
            # inference=mac
        )
        print(assignment)

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
        print('Games from:', student)
        try_csps(csps)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')