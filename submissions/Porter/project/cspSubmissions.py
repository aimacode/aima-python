import importlib
import traceback
from util import roster, print_table
from csp import backtracking_search, lcv, mrv, mac
import myCSPs
# import tkinter
# from Porter.Output import sudokuGUI
# gui = sudokuGUI.initialize(sudokuGUI(tkinter))
# states = {}
    # 'A1': 3, 'A3': 1, 'A4': 5, 'A5': 2, 'A6': 9,
#                    'B1': 9, 'B3': 4, 'B7': 3, 'B8': 5,
#                    'C5': 3, 'C8': 8,
#                    'D1': 1, 'D2': 2, 'D3': 5, 'D4': 3, 'D5': 8,
#                    'E4': 1, 'E5': 4, 'E7': 7, 'E9': 3,
#                    'F1': 7, 'F9': 5,
#                    'G1': 8, 'G6': 3, 'G8': 9,
#                    'H2': 1, 'H5': 7, 'H9': 8,
#                    'I1': 5, 'I2': 3, 'I3': 9, 'I4': 2, 'I5': 1, 'I6': 8, 'I8': 7, 'I9': 6}

def try_csps(csps):

    for c in csps:
        # myCSPs.eliminateVariables(state)
        assignment = backtracking_search(
            **c,

            order_domain_values=lcv,
            select_unassigned_variable = mrv,
            inference=mac
        )
        # print(assignment)
        final = assignment
        return final


submissions = {}
scores = {}

message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module(student + '.myCSPs')
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


