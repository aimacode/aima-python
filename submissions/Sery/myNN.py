from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Sery import aids

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

aidsECHP = DataFrame()
aidsECHP.data = []
target_data = []

list_of_report = aids.get_reports()
for record in list_of_report:
    try:
        prevalence = float(record['Data']["HIV Prevalence"]["Adults"])
        target_data.append(prevalence)

        year = int(record['Year'])
        living =  int(record['Data']["People Living with HIV"]["Adults"])
        new = int(record['Data']["New HIV Infections"]["Adults"])
        deaths = int(record['Data']["AIDS-Related Deaths"]["Adults"])

        aidsECHP.data.append([year, living, new, deaths])

    except:
        traceback.print_exc()

aidsECHP.feature_names = [
    'Year',
    'People Living with HIV',
    'New HIV Infections',
    'AIDS-Related Deaths',
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
aidsECHP.target = []

def aidsTarget(percentage):
    if percentage > 6:
        return 1
    return 0

for pre in target_data:
    # choose the target
    tt = aidsTarget(pre)
    aidsECHP.target.append(tt)

aidsECHP.target_names = [
    'HIV Prevalence <= 6%',
    'HIV Prevalence > 6%',
]

'''
Make a customn classifier,
'''
mlpc = MLPClassifier(
    hidden_layer_sizes = (120,),
    activation = 'relu',
    solver='sgd', # 'adam',
    alpha = 0.00001,
    # batch_size='auto',
    learning_rate = 'adaptive', # 'constant',
    # power_t = 0.5,
    max_iter = 1200, # 200,
    shuffle = True,
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
aidsScaled = DataFrame()

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

setupScales(aidsECHP.data)
aidsScaled.data = scaleGrid(aidsECHP.data)
aidsScaled.feature_names = aidsECHP.feature_names
aidsScaled.target = aidsECHP.target
aidsScaled.target_names = aidsECHP.target_names

Examples = {
    'AidsDefault': {
        'frame': aidsECHP,
    },
    'AidsSGD': {
        'frame': aidsECHP,
        'mlpc': mlpc
    },
    'AidsScaled': {
        'frame': aidsScaled,
    },
}