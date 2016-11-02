import traceback

from submissions.Miles import food



class DataFrame:
    data = []  # grid of floating point numbers
    feature_names = []  # column names
    target = []  # list of the target value
    target_names = []  # target labels

foodFF = DataFrame()  # what I pass on to examples

'''
Extract data from the CORGIS food.
'''
joint = {}

food = food.get_reports()
for category in food:
    try:
        it = category['Category'] # ['Carbohydrate']
        vit = category['Data']['Vitamins']['Vitamin A - IU']
        joint[it] = {}
        joint[it]['IT']= it
        joint[it]['Data'] = vit
    except:
        traceback.print_exc()
#
# demographics = county_demographics.get_all_counties()
for county2 in food:
    try:
        # countyNames = county['County'].split()
        # cName = ' '.join(countyNames[:-1])
        # st = county['State']
        # countyST = cName + st
        carbs = county2["Data"]# ["Carbohydrate"]
        fiber = county2['Data']["Fiber"]
        NDBN = county2['Data'] # ["Nutrient Data Bank Number"]
        vitamins = county2['Data']["Vitamins"]["Vitamin A - IU"]
        if it in joint:
            joint[it]['Carbohydrates'] = carbs
            joint[it]['Fiber'] = fiber
            joint[it]['Nutrient Data Bank Number'] = NDBN
            joint[it]['Vitamin A'] = vitamins
    except:
        traceback.print_exc()
#
#
#
foodFF.data = []
#
# '''
# Build the input frame, row by row.
# '''
for it in joint:
    # choose the input values
    foodFF.data.append([
        it,
        joint[it]['IT'],
        # joint[it]['Category'],
        # joint[it]['Carbohydrates'],
        # joint[it]['Vitamin A'],
        # joint[it]['Fiber'],
        # joint[it]['Nutrient Data Bank Number'],
    ])

foodFF.feature_names = [
    'it',
    'IT',
    'Category',
    'Carbohydrates',
    'Vitamin A',
    'Fiber',
    'Nutrient Data Bank Number',
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

Examples = {
    'Food': foodFF,
}