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

for fiber in food:
    try:
        foodIT = fiber["Data"]["Fiber"]
        joint[foodIT] = {}
        # joint[it]['IT']= it
        # joint[it]['Data'] = vit
    except:
        traceback.print_exc()

# demographics = county_demographics.get_all_counties()
for item in food:
    try:
        # countyNames = county['County'].split()
        # cName = ' '.join(countyNames[:-1])
        # st = county['State']
        # countyST = cName + st
        # foodIT = item["Data"]["Carbohydrate"]
        category = item["Category"]
        description = item["Description"]
        NDBN = item["Nutrient Data Bank Number"]
        # vitamins = county2['Data']["Vitamins"]["Vitamin A - IU"]
        # if foodIT in joint:
        #     joint[foodIT]["Category"] = category
        #     joint[foodIT]["Description"] = description
        #     joint[foodIT]["Nutrient Data Bank Number"] = NDBN
        #    # joint[foodIT]['Vitamin A'] = vitamins
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
# for foodIT in joint:
#     # choose the input values
#     foodFF.data.append([
#         foodIT,
#         # joint[foodIT]['Fiber'],
#         joint[foodIT]['Category'],
#         joint[foodIT]['Description'],
#         joint[foodIT]['Nutrient Data Bank Number'],
#
#     ])

foodFF.feature_names = [

    'Fiber',
    'Category',
    'Description',
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
foodFF.target = []
#
def foodTarget(grams):
    if grams > 20:
        return 1
    return 0

# for countyST in intersection:
#     # choose the target
#     tt = trumpTarget(intersection[countyST]['Trump'])
#     trumpECHP.target.append(tt)
#
foodFF.target_names = [
    'Fiber <= 20',
    'Fiber >  20',
]

Examples = {
    'Food': foodFF,
}