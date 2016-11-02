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

    # if grams.__contains__('B'):
    #     return 1
    # return 0

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

