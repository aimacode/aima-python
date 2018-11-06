import importlib
import traceback
from grading.util import roster, print_table

def indent(howMuch = 1):
    space = ' '
    for i in range(1, howMuch):
        space += '  '
    return space

def tryOne(label, example):
    try:
        data, target, model, weights = example
    except:
        print('Error:', example, 'should have 4 elements.')
        return null
    try:
        model.fit(data, target, epochs=1)
    except:
        print('Error: model.fit(data, target, epochs=1) fails.')
        return null
    try:
        model.set_weights(weights)
    except:
        print('Error: model.set_weights(weights) fails.')
        return null
    try:
        raw = model.predict(data)
    except:
        print('Error: model.predict(data) fails.')
        return null
    prediction = raw.round()
    print(label + ':')
    # print_table(fit.theta_,
    #             header=[frame.feature_names],
    #             topLeft=[label],
    #             leftColumn=frame.target_names,
    #             numfmt='%6.3f',
    #             njust='center',
    #             tjust='rjust',
    #             )
    tot = prediction.size
    mis = (target != prediction).sum()
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

# root = os.getcwd()
for student in roster:
    try:
        # os.chdir(root + '/submissions/' + student)
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myNN')
        submissions[student] = mod.Examples
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

# os.chdir(root)

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
