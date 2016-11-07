import traceback
from submissions.Haller import school_scores

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

mathSATScores = DataFrame()

'''
Extract data from the CORGIS School Scores.
'''
joint = {}

scores = school_scores.get_all()
for state in scores:
    try:
        st = state['State']['Code']
        totalNum = state['Total']['Test-takers']
        totalAvgMath = state['Total']['Math']
        if totalNum != 0:
            malePct = state['Gender']['Male']['Test-takers'] / totalNum * 100
            femalePct = state['Gender']['Female']['Test-takers'] / totalNum * 100
            aPlusPct = state['GPA']['A plus']['Test-takers'] / totalNum * 100
            aMinusPct = state['GPA']['A minus']['Test-takers'] / totalNum * 100
            aPct = state['GPA']['A']['Test-takers'] / totalNum * 100
            bPct = state['GPA']['B']['Test-takers'] / totalNum * 100
        else:
            malePct = 0
            femalePct = 0
            aPlusPct = 0
            aMinusPct = 0
            aPct = 0
            bPct = 0
        joint[st] = {}
        joint[st]['ST']= st
        joint[st]['Male Percent'] = malePct
        joint[st]['Female Percent'] = femalePct
        joint[st]['A+ Percent'] = aPlusPct
        joint[st]['A Percent'] = aPct
        joint[st]['A- Percent'] = aMinusPct
        joint[st]['B Percent'] = bPct
        joint[st]['Average Math SAT'] = totalAvgMath
    except:
        traceback.print_exc()

mathSATScores.data = []

'''
Build the input frame, row by row.
'''
for state in joint:
    # choose the input values
    mathSATScores.data.append([
        joint[state]['Male Percent'],
        joint[state]['Female Percent'],
        joint[state]['A+ Percent'],
        joint[state]['A- Percent'],
        joint[state]['A Percent'],
        joint[state]['B Percent'],
    ])

mathSATScores.feature_names = [
    'Male Percent',
    'Female Percent',
    'A+ Percent',
    'A- Percent',
    'A Percent',
    'B Percent',
]

'''
Build the target list,
one entry for each row in the input frame.

The Naive Bayesian network is a classifier,
i.e. it sorts data points into bins.
The best it can do to estimate a continuous variable
is to break the domain into segments, and predict
the segment into which the variable's value will fall.
In this example, I'm breaking Trump's % into two
arbitrary segments.
'''
mathSATScores.target = []

def SATTarget(number):
    if number > 550:
        return 1
    return 0

for state in joint:
    # choose the target
    sTar = SATTarget(joint[state]['Average Math SAT'])
    mathSATScores.target.append(sTar)

mathSATScores.target_names = [
    'Average Math SAT <= 550',
    'Average Math SAT > 550',
]

Examples = {
    'Average Math SAT': mathSATScores,
}