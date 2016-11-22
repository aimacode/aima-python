import importlib
import traceback
from grading.util import roster, print_table
# from logic import FolKB
# from utils import expr
import os
from sklearn.neural_network import MLPClassifier

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
        clf = MLPClassifier()
    try:
        fit = clf.fit(frame.data, frame.target)
    except:
        traceback.print_exc()
    print(label + ':')
    # print_table(fit.theta_,
    #             header=[frame.feature_names],
    #             topLeft=[label],
    #             leftColumn=frame.target_names,
    #             numfmt='%6.3f',
    #             njust='center',
    #             tjust='rjust',
    #             )
    y_pred = fit.predict(frame.data)
    tot = len(frame.data)
    mis = (frame.target != y_pred).sum()
    cor = 1 - mis / tot
    print(
        "  Number of mislabeled points out of a total {0} points : {1} ({2:.0%} correct)"
            .format(tot, mis, cor)
    )

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
        print('Neural Networks from:', student)
        tryExamples(examples)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')