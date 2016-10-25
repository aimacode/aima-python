import traceback

from submissions.aartiste import election
from submissions.aartiste import county_demographics

elections = election.get_results()
demographics = county_demographics.get_all_counties()

frame = {}
frame['feature_names'] = [
    'CountyST', # unique identifier, County + ST
    'ST', 'Trump', # from elections
    'College', 'Poverty', # from demographics
]

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
        countST = cName + st
        college = county['Education']["Bachelor's Degree or Higher"]
        poverty = county['Income']["Persons Below Poverty Level"]
        joint[countyST]['College'] = college
    except:
        traceback.print_exc()


frame['data'] = []

frame['target'] = []

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(frame.data, frame.target).predict(frame.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (frame.data.shape[0],(frame.target != y_pred).sum()))