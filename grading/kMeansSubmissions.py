import importlib
import traceback
from grading.util import roster, print_table
import os
from sklearn.cluster import KMeans
from sklearn import metrics
from numpy import unique
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def plot(title, X3D):
#     fig = plt.figure(title)
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#     pass

def indent(howMuch = 1):
    space = ' '
    for i in range(1, howMuch):
        space += '  '
    return space

def tryOne(label, kDict):
    data = kDict['data']
    kValues = kDict['k']
    if 'main' in kDict:
        main = kDict['main']
        try:
            main()
        except:
            traceback.print_exc()
    for nc in kValues:
        print('%s[%d]:' % (label, nc))
        kmeans = KMeans(n_clusters=nc)
        try:
            kmeans.fit(data)
        except:
            traceback.print_exc()
            continue
        print(kmeans.cluster_centers_)
        labels = kmeans.labels_

        # https://sklearn.org/modules/clustering.html#silhouette-coefficient
        sscore = metrics.silhouette_score(data, labels)
        print('Silhouette Coefficient: %f' % sscore)

        # https://sklearn.org/modules/clustering.html#calinski-harabaz-index
        chindex = metrics.calinski_harabaz_score(data, labels)
        print('Calinski-Harabaz Index: %f' % chindex)
        # print(kmeans.get_params())
    # score = metrics.adjusted_rand_score(frame.target, fit.labels_)
    # print('Adjusted Rand index: ', score)

def tryExamples(examples):
    for label in examples:
        example = examples[label]
        tryOne(label, example)

submissions = {}
scores = {}

message1 = 'Submissions that compile:'

root = os.getcwd()
for student in roster:
    try:
        directory = root + '/../submissions/' + student
        os.chdir(directory)
    except:
        print('missing directory: ' + directory)
        continue
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myKMeans')
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
        print('K Means Samples from:', student)
        tryExamples(examples)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')
