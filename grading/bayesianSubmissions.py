import importlib
import traceback
from probability import BayesNet, enumeration_ask, enumerate_all, elimination_ask

from grading.util import roster, print_table
# from logic import FolKB
# from utils import expr
import os
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()

def indent(howMuch = 1):
    space = ' '
    for i in range(1, howMuch):
        space += '  '
    return space

# def printKB(label, kb):
#     print(indent(), label + ' example:')
#     print(indent(2), 'knowledge base:')
#     for clause in kb.clauses:
#         print(indent(3), str(clause))

def printResults(query, gen, limit=3):
    for count in range(limit):
        try:
            long = next(gen)
        except StopIteration:
            print()
            return
        short = {}
        for v in long:
            if v in query.args:
                short[v] = long[v]
        print(short, end=' ')
    print('...')

def printNode(node):
    print('(\'' + node.variable + '\',', str(node.parents))
    print(' ', str(node.cpt) + '),')

def printBN(bn):
    print(bn.label + ':')
    for node in bn.nodes:
        printNode(node)


def tryOne(bn, query):
    X = query['variable']
    e = query['evidence']
    estr = str(e)[1:-1]
    print('P( %s | %s ) = ' % (X, estr), end='')
    prob = elimination_ask(X, e, bn)
    print(prob.show_approx())

def tryExamples(examples):
    for bn in examples:
        printBN(bn)
        qlist = examples[bn]
        for query in qlist:
            tryOne(bn, query)

submissions = {}
scores = {}

message1 = 'Submissions that compile:'

root = os.getcwd()
for student in roster:
    try:
        # os.chdir(root + '/submissions/' + student)
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myBayes')
        submissions[student] = mod.examples
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

    print(student, 'summary:', str(scores[student]), '\n' +
          student, '  total:', str(sum(scores[student])), '\n' +
          '----------------------------------------')
