from sklearn.cluster import KMeans
import traceback
from submissions.Miles import food
from sklearn import datasets


# class DataFrame:
#     data = []
#     feature_names = []
#     target = []
#     target_names = []
#
# trumpECHP = DataFrame()
#
# '''
# Extract data from the CORGIS elections, and merge it with the
# CORGIS demographics.  Both data sets are organized by county and state.
# '''
# joint = {}
#
# elections = election.get_results()
# for county in elections:
#     try:
#         st = county['Location']['State Abbreviation']
#         countyST = county['Location']['County'] + st
#         trump = county['Vote Data']['Donald Trump']['Percent of Votes']
#         joint[countyST] = {}
#         joint[countyST]['ST']= st
#         joint[countyST]['Trump'] = trump
#     except:
#         traceback.print_exc()
#
# demographics = county_demographics.get_all_counties()
# for county in demographics:
#     try:
#         countyNames = county['County'].split()
#         cName = ' '.join(countyNames[:-1])
#         st = county['State']
#         countyST = cName + st
#         # elderly =
#         # college =
#         # home =
#         # poverty =
#         if countyST in joint:
#             joint[countyST]['Elderly'] = county['Age']["Percent 65 and Older"]
#             joint[countyST]['HighSchool'] = county['Education']["High School or Higher"]
#             joint[countyST]['College'] = county['Education']["Bachelor's Degree or Higher"]
#             joint[countyST]['White'] = county['Ethnicities']["White Alone, not Hispanic or Latino"]
#             joint[countyST]['Persons'] = county['Housing']["Persons per Household"]
#             joint[countyST]['Home'] = county['Housing']["Homeownership Rate"]
#             joint[countyST]['Income'] = county['Income']["Median Houseold Income"]
#             joint[countyST]['Poverty'] = county['Income']["Persons Below Poverty Level"]
#             joint[countyST]['Sales'] = county['Sales']["Retail Sales per Capita"]
#     except:
#         traceback.print_exc()
#
# '''
# Remove the counties that did not appear in both samples.
# '''
# intersection = {}
# for countyST in joint:
#     if 'College' in joint[countyST]:
#         intersection[countyST] = joint[countyST]
#
# trumpECHP.data = []
#
# '''
# Build the input frame, row by row.
# '''
# for countyST in intersection:
#     # choose the input values
#     row = []
#     for key in intersection[countyST]:
#         if key in ['ST', 'Trump']:
#             continue
#         row.append(intersection[countyST][key])
#     trumpECHP.data.append(row)
#
# firstCounty = next(iter(intersection.keys()))
# firstRow = intersection[firstCounty]
# trumpECHP.feature_names = list(firstRow.keys())
# trumpECHP.feature_names.remove('ST')
# trumpECHP.feature_names.remove('Trump')
#
# '''
# Build the target list,
# one entry for each row in the input frame.
#
# The Naive Bayesian network is a classifier,
# i.e. it sorts data points into bins.
# The best it can do to estimate a continuous variable
# is to break the domain into segments, and predict
# the segment into which the variable's value will fall.
# In this example, I'm breaking Trump's % into two
# arbitrary segments.
# '''
# trumpECHP.target = []
#
# def trumpTarget(percentage):
#     if percentage > 45:
#         return 1
#     return 0
#
# for countyST in intersection:
#     # choose the target
#     tt = trumpTarget(intersection[countyST]['Trump'])
#     trumpECHP.target.append(tt)
#
# trumpECHP.target_names = [
#     'Trump <= 45%',
#     'Trump >  45%',
# ]

class DataFrame:
    data = []  # grid of floating point numbers
    feature_names = []  # column names
    target = []  # list of the target value
    target_names = []  # target labels



foodFF = DataFrame()  # what I pass on to examples
foodFF.data = []
targetData = []

'''
Extract data from the CORGIS food.
'''


food = food.get_reports()

for info in food:
    try:
        # item = str(info["Category"])
        item = float(info["Data"]["Fat"]["Saturated Fat"])
        targetData.append(item)

        fiber = float(info["Data"]["Fiber"])
        carbohydrate = float(info["Data"]["Carboydrate"]) # they misspelled carbohydrates LOL
        water = float(info["Data"]["Water"])
        vitamin = float(info["Data"]["Vitamins"]["Vitamin C"])

        foodFF.data.append([fiber, carbohydrate, water, vitamin])

    except:
        traceback.print_exc()


foodFF.feature_names = [


    'Fiber',
    'Carbohydrates',
    'Water',
    'Vitamin C',
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

foodFF.target = []



def foodTarget(percentage):

    if percentage > 10:
        return 1
    return 0



for item2 in targetData:
    # choose the target
    target_t = foodTarget(item2)
    foodFF.target.append(target_t)
# comparing the fat contents of a food to other contents of same food
foodFF.target_names = [
    'Saturated Fat is <= 10%',
    'Saturated Fat is > 10%',
    # 'Butter',
    # 'Milk',
    # 'Cheese'
]

Examples = {
    'Food': foodFF,
}



'''
Try scaling the data.
'''
foodScaled = DataFrame()

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

setupScales(foodFF.data)
foodScaled.data = scaleGrid(foodFF.data)
foodScaled.feature_names = foodFF.feature_names
foodScaled.target = foodFF.target
foodScaled.target_names = foodFF.target_names

'''
Make a customn classifier,
'''
km = KMeans(
    n_clusters=8,
    max_iter=300,
    n_init=10,
    init='k-means++',
    algorithm='auto',
    precompute_distances='auto',
    tol=1e-4,
    n_jobs=-1,
    # random_state= numpy.random.RandomState,
    verbose=0,
    copy_x=True,
)

Examples = {
    'Food': {
        'frame': foodScaled,
    },
    'FoodCustom': {
        'frame': foodScaled,
        'kmeans': km
    },
}