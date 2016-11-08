from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Fritz import medal_of_honor

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

honordata = DataFrame()
honordata.data = []
honortarget = []


medalofhonor = medal_of_honor.get_awardees(test=True)
for issued in medalofhonor:
    try:
        date = int(issued['birth']["date"]["year"])
        honortarget.append(date)


        day = int(issued['awarded']['date']['day'])
        month = int(issued['awarded']['date']['month'])
        year = int(issued['awarded']['date']['year'])


        honordata.data.append([day, month, year])

    except:
        traceback.print_exc()

honordata.feature_names = [
    'day',
    'month',
    'year',
]


honordata.target = []

def targetdata(HDate):
    if (HDate > 1880 and HDate != -1):
        return 1
    return 0


for issued in honortarget:

    TD = targetdata(issued)
    honordata.target.append(TD)

honordata.target_names = [
    'Born before 1880',
    'Born after 1880',
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
dateScaled = DataFrame()

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

setupScales(honordata.data)
dateScaled.data = scaleGrid(honordata.data)
dateScaled.feature_names = honordata.feature_names
dateScaled.target = honordata.target
dateScaled.target_names = honordata.target_names

Examples = {
    'Default Date':{
    'frame': honordata,
    },
    'DateSGD': {
        'frame': honordata,
        'mlpc': mlpc
    },
    'dateScaled': {
        'frame': dateScaled,
    },
}
