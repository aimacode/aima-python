from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.aartiste import election
from submissions.aartiste import county_demographics

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

trumpECHP = DataFrame()

'''
Extract data from the CORGIS elections, and merge it with the
CORGIS demographics.  Both data sets are organized by county and state.
'''
joint = {}

elections = election.get_results()
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

demographics = county_demographics.get_all_counties()
for county in demographics:
    try:
        countyNames = county['County'].split()
        cName = ' '.join(countyNames[:-1])
        st = county['State']
        countyST = cName + st
        # elderly =
        # college =
        # home =
        # poverty =
        if countyST in joint:
            joint[countyST]['Elderly'] = county['Age']["Percent 65 and Older"]
            joint[countyST]['HighSchool'] = county['Education']["High School or Higher"]
            joint[countyST]['College'] = county['Education']["Bachelor's Degree or Higher"]
            joint[countyST]['White'] = county['Ethnicities']["White Alone, not Hispanic or Latino"]
            joint[countyST]['Persons'] = county['Housing']["Persons per Household"]
            joint[countyST]['Home'] = county['Housing']["Homeownership Rate"]
            joint[countyST]['Income'] = county['Income']["Median Houseold Income"]
            joint[countyST]['Poverty'] = county['Income']["Persons Below Poverty Level"]
            joint[countyST]['Sales'] = county['Sales']["Retail Sales per Capita"]
    except:
        traceback.print_exc()

'''
Remove the counties that did not appear in both samples.
'''
intersection = {}
for countyST in joint:
    if 'College' in joint[countyST]:
        intersection[countyST] = joint[countyST]

trumpECHP.data = []

'''
Build the input frame, row by row.
'''
for countyST in intersection:
    # choose the input values
    row = []
    for key in intersection[countyST]:
        if key in ['ST', 'Trump']:
            continue
        row.append(intersection[countyST][key])
    trumpECHP.data.append(row)

firstCounty = next(iter(intersection.keys()))
firstRow = intersection[firstCounty]
trumpECHP.feature_names = list(firstRow.keys())
trumpECHP.feature_names.remove('ST')
trumpECHP.feature_names.remove('Trump')

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
trumpECHP.target = []

def trumpTarget(percentage):
    if percentage > 45:
        return 1
    return 0

for countyST in intersection:
    # choose the target
    tt = trumpTarget(intersection[countyST]['Trump'])
    trumpECHP.target.append(tt)

trumpECHP.target_names = [
    'Trump <= 45%',
    'Trump >  45%',
]

'''
Make a customn classifier,
'''
mlpc = MLPClassifier(
    # hidden_layer_sizes = (100,),
    # activation = 'relu',
    solver='sgd', # 'adam',
    # alpha = 0.0001,
    # batch_size='auto',
    learning_rate = 'adaptive', # 'constant',
    # power_t = 0.5,
    max_iter = 1000, # 200,
    # shuffle = True,
    # random_state = None,
    # tol = 1e-4,
    # verbose = False,
    # warm_start = False,
    # momentum = 0.9,
    # nesterovs_momentum = True,
    # early_stopping = False,
    # validation_fraction = 0.1,
    # beta_1 = 0.9,
    # beta_2 = 0.999,
    # epsilon = 1e-8,
)

'''
Try scaling the data.
'''
trumpScaled = DataFrame()

def setupScales(grid):
    global min, max
    min = list(grid[0])
    max = list(grid[0])
    for row in range(1, len(grid)):
        for col in range(len(grid[row])):
            cell = grid[row][col]
            if cell < min[col]:
                min[col] = cell
            if cell > max[col]:
                max[col] = cell

def scaleGrid(grid):
    newGrid = []
    for row in range(len(grid)):
        newRow = []
        for col in range(len(grid[row])):
            try:
                cell = grid[row][col]
                scaled = (cell - min[col]) \
                         / (max[col] - min[col])
                newRow.append(scaled)
            except:
                pass
        newGrid.append(newRow)
    return newGrid

setupScales(trumpECHP.data)
trumpScaled.data = scaleGrid(trumpECHP.data)
trumpScaled.feature_names = trumpECHP.feature_names
trumpScaled.target = trumpECHP.target
trumpScaled.target_names = trumpECHP.target_names

Examples = {
    'TrumpDefault': {
        'frame': trumpECHP,
    },
    'TrumpSGD': {
        'frame': trumpECHP,
        'mlpc': mlpc
    },
    'TrumpScaled': {
        'frame': trumpScaled,
    },
}