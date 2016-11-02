import traceback

from submissions.Miles import food



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
        item = float(info["Data"]["Fat"]["Saturated Fat"])
        targetData.append(item)

        fiber = float(info["Data"]["Fiber"])
        carbohydrate = float(info["Data"]["Carboydrate"]) # they misspelled carbohydrates LOL
        water = float(info["Data"]["Water"])

        foodFF.data.append([fiber, carbohydrate, water])

    except:
        traceback.print_exc()


foodFF.feature_names = [


    'Fiber',
    'Carbohydrates',
    'water',
]
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
foodFF.target = []
#


def foodTarget(percentage):


    if percentage > 2:
        return 1
    return 0

for pre in targetData:
    # choose the target
    tt = foodTarget(pre)
    foodFF.target.append(tt)
# for countyST in intersection:
#     # choose the target
#     tt = trumpTarget(intersection[countyST]['Trump'])
#     trumpECHP.target.append(tt)
#
foodFF.target_names = [
    'Fat <= 2%',
    'Fat >  2%',
]

Examples = {
    'Food': foodFF,
}

