import importlib
import traceback
from grading.util import roster, print_table
from logic import FolKB
from utils import expr

def indent(howMuch = 1):
    space = ' '
    for i in range(1, howMuch):
        space += '  '
    return space

def printKB(label, kb):
    print(indent(), label + ' example:')
    print(indent(2), 'knowledge base:')
    for clause in kb.clauses:
        print(indent(3), str(clause))

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

def tryKB(label, base):
    kbString = base['kb']
    kb = FolKB([])
    for kbLine in kbString.split('\n'):
        s = kbLine.strip()
        if len(s) > 0:
            try:
                sentence = expr(s)
                kb.tell(sentence)
            except:
                traceback.print_exc()
    printKB(label, kb)
    print(indent(2), 'queries:')
    queryString = base['queries']
    for qLine in queryString.split('\n'):
        s = qLine.strip()
        if len(s) > 0:
            try:
                query = expr(s)
                generator = kb.ask_generator(query)
                print(indent(3), str(query) + '?', end=' ')
                if 'limit' in base:
                    printResults(query, generator, base['limit'])
                else:
                    printResults(query, generator)
            except:
                traceback.print_exc()

def try_kbs(bases):
    for label in bases:
        tryKB(label, bases[label])

submissions = {}
scores = {}

message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.myLogic')
        submissions[student] = mod.Examples
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
        bases = submissions[student]
        print('Knowledge Bases from:', student)
        try_kbs(bases)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')