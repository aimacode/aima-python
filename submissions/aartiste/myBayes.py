import traceback

from submissions.aartiste import election
from submissions.aartiste import county_demographics

elections = election.get_results()
demographics = county_demographics.get_all_counties()

class DataFrame:
    feature_names = []
    data = []
    target = []

trumpECHP = DataFrame()

joint = {}
for county in elections:
    try:
        st = county['Location']['State Abbreviation']
        countyST = county['Location']['County'] + st
        trump = county['Vote Data']['Donald Trump']['Percent of Votes']
        joint[countyST] = {}
        joint[countyST]['ST']= st
        joint[countyST]['Trump'] = trump
    except:
        traceback.print_exc()

for county in demographics:
    try:
        countyNames = county['County'].split()
        cName = ' '.join(countyNames[:-1])
        st = county['State']
        countyST = cName + st
        elderly = county['Age']["Percent 65 and Older"]
        college = county['Education']["Bachelor's Degree or Higher"]
        home = county['Housing']["Homeownership Rate"]
        poverty = county['Income']["Persons Below Poverty Level"]
        if countyST in joint:
            joint[countyST]['Elderly'] = elderly
            joint[countyST]['College'] = college
            joint[countyST]['Home'] = home
            joint[countyST]['Poverty'] = poverty
    except:
        traceback.print_exc()

intersection = {}
for countyST in joint:
    if 'College' in joint[countyST]:
        intersection[countyST] = joint[countyST]

trumpECHP.data = []
trumpECHP.target = []

'''
The Naive Bayesian network is a classifier,
i.e. it sorts data points into bins.
The best it can do to estimate a continuous variable
is to break the domain into segments, and predict
the segment into which the variable's value will fall.
In this example, I'm breaking Trump's % into two
arbitrary segments.
'''
def trumpTarget(percentage):
    if percentage > 45:
        return 1
    return 0

# Build the input frame and the target array,
# row by row.
for countyST in intersection:
    # choose the input values
    trumpECHP.data.append([
        # countyST,
        # intersection[countyST]['ST'],
        # intersection[countyST]['Trump'],
        intersection[countyST]['Elderly'],
        intersection[countyST]['College'],
        intersection[countyST]['Home'],
        intersection[countyST]['Poverty'],
    ])
    # choose the target
    tt = trumpTarget(intersection[countyST]['Trump'])
    trumpECHP.target.append(tt)

trumpECHP.feature_names = {
    'data': [
        # 'countyST',
        # 'ST',
        # 'Trump',
        'Elderly',
        'College',
        'Home',
        'Poverty'
    ],
    'target': 'Trump'
}

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# fit = gnb.fit(trumpECHP.data, trumpECHP.target)
# y_pred = fit.predict(trumpECHP.data)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (len(trumpECHP.data), (trumpECHP.target != y_pred).sum()))

Examples = {
    'Trump': trumpECHP,
}