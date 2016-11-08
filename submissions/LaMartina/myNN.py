import traceback
from submissions.LaMartina import state_crime


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

crimes = DataFrame()

'''
Extract data from the CORGIS state_crime.
'''
joint = {}

crime = state_crime.get_all_crimes()

for c in crime:
    try:
        #if c['State'] == 'Alabama':
        stateyear = c['State'] + str(c['Year'])
        pop = c['Data']['Population']
        murder = c['Data']['Totals']['Violent']['Murder']
        rape = c['Data']['Totals']['Violent']['Rape']
        burg = c['Data']['Totals']['Property']['Burglary']
        joint[stateyear] = {}
        joint[stateyear]['Population'] = pop
        joint[stateyear]['Murder Numbers'] = murder
        joint[stateyear]['Rape Numbers'] = rape
        joint[stateyear]['Burglary Numbers'] = burg
    except:
        traceback.print_exc()

crimes.data = []

'''
Build the input frame, row by row.
'''
for port in joint:
    # choose the input values
    crimes.data.append([
        #port,
        joint[port]['Population'],
        joint[port]['Rape Numbers'],
        joint[port]['Burglary Numbers'],
    ])
crimes.feature_names = [
    'Population',
    'Rape Numbers',
    'Burglary Numbers',
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
crimes.target = []

def murderTarget(murdernum):
    if murdernum > 800:
        return 1
    return 0

for cri in joint:
    # choose the target
    c = murderTarget(joint[cri]['Murder Numbers'])
    crimes.target.append(c)

crimes.target_names = [
    'Murders <= 800',
    'Murders >  800',
]

Examples = {
    'Crimes': {
        'frame': crimes,
    },
}