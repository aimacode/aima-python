from sklearn.cluster import KMeans
import traceback
from submissions.Miles import food
from sklearn import datasets




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
    n_clusters=12,
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