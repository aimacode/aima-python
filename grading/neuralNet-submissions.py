import importlib
import traceback
from grading.util import roster, print_table
# from logic import FolKB
# from utils import expr
import os
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier()

def indent(howMuch = 1):
    space = ' '
    for i in range(1, howMuch):
        space += '  '
    return space

def tryOne(label, fAndP):
    frame = fAndP['frame']
    if 'mlpc' in fAndP.keys():
        clf = fAndP['mlpc']
    else:
        clf = mlpc
    fit = clf.fit(frame.data, frame.target)
    print('')
    # print_table(fit.theta_,
    #             header=[frame.feature_names],
    #             topLeft=[label],
    #             leftColumn=frame.target_names,
    #             numfmt='%6.3f',
    #             njust='center',
    #             tjust='rjust',
    #             )
    y_pred = fit.predict(frame.data)
    print("Number of mislabeled points out of a total %d points : %d"
          % (len(frame.data), (frame.target != y_pred).sum()))

def tryExamples(examples):
    for label in examples:
        example = examples[label]
        main = getattr(example, 'main', None)
        if main != None:
            example.main()
        else:
            tryOne(label, example)

submissions = {}
scores = {}

message1 = 'Submissions that compile:'

root = os.getcwd()
for student in roster:
    try:
        os.chdir(root + '/submissions/' + student)
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myNN')
        submissions[student] = mod.Examples
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

os.chdir(root)

print(message1)
print('----------------------------------------')

for student in roster:
    if not student in submissions.keys():
        continue
    scores[student] = []
    try:
        examples = submissions[student]
        print('Bayesian Networks from:', student)
        tryExamples(examples)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')