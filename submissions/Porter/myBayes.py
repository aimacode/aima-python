import traceback

from submissions.Porter import billionaires



class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

billionairesECHP = DataFrame()



billionaireInfo = billionaires.get_billionaires()


'''
Build the input frame, row by row.
'''
for countyST in intersection:
    # choose the input values
    billionairesECHP5.data.append([
        # countyST,
        # intersection[countyST]['ST'],
        # intersection[countyST]['Trump'],
        intersection[countyST]['Elderly'],
        intersection[countyST]['College'],
        intersection[countyST]['Home'],
        intersection[countyST]['Poverty'],
    ])

billionairesECHP.feature_names = [
    # 'countyST',
    # 'ST',
    # 'Trump',
    'Political',
    'Inherited',
    'Gender',

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
billionairesECHP.target = []

def trumpTarget(percentage):
    if percentage > 45:
        return 1
    return 0

for countyST in intersection:
    # choose the target
    tt = trumpTarget(intersection[countyST]['Trump'])
    trumpECHP.target.append(tt)

billionairesECHP.target_names = [
    'New',
    'Old',
]

Examples = {
    'Billionaires': billionairesECHP,
}