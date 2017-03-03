from sklearn.cluster import KMeans
import traceback
from sklearn.neural_network import MLPClassifier
from submissions.Hess import cars

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

guzzle = DataFrame()
guzzle.target = []
guzzle.data = []

guzzle = cars.get_cars()

def guzzleTarget(string):
    if (info['Fuel Information']['City mph'] < 14):
        return 1
    return 0

for info in guzzle:
    try:

        guzzle.data.append(guzzleTarget(info['Fuel Information']['City mph']))
        fuelCity = float(info['Fuel Information']['City mph']) # they misspelled mpg
        year = float(info['Identification']['Year'])

        guzzle.data.apend([fuelCity, year])
    except:

        traceback.print_exc()

guzzle.feature_names = [
    "City mph"
    "Year"
]

guzzle.target_names = [
    "New Car is < 14 MPG"
    "New Car is > 14 MPG"
]

mlpc = MLPClassifier(
    solver='sgd',
    learning_rate = 'adaptive',
)

'''
Try scaling the data.
'''
guzzleScaled = DataFrame()

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

#custom classifier

km = KMeans(
n_clusters=2,
)

Examples = {
    'Guzzle': {
        'frame': guzzleScaled,
        'kmeans': km
    }
    # 'GuzzleMLPC': {
    #     'frame': guzzle,
    #     'mlpc': mlpc
    # },
}
