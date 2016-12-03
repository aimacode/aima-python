from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Kinley import drugs

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

drugData = DataFrame()
drugData.data = []
targetData = []

alcohol = drugs.get_surveys('Alcohol Dependence')
#tobacco = drugs.get_surveys('Tobacco Use')

i=0

for survey in alcohol[0]['data']:
    try:
        youngUser = float(survey['Young']),
        youngUserFloat = youngUser[0]
        midUser = float(survey['Medium']),
        midUserFloat = midUser[0]
        oldUser = float(survey['Old']),
        oldUserFloat = oldUser[0]
        place = survey['State']

        total = youngUserFloat + midUserFloat + oldUserFloat
        targetData.append(total)

        youngCertain = float(survey['Young CI']),
        youngCertainFloat = youngCertain[0]
        midCertain = float(survey['Medium CI']),
        midCertainFloat = midCertain[0]
        oldCertain = float(survey['Old CI']),
        oldCertainFloat = oldCertain[0]

        drugData.data.append([youngCertainFloat, midCertainFloat, oldCertainFloat])

        i = i + 1
    except:
        traceback.print_exc()


drugData.feature_names = [

      'Young CI',
      'Medium CI',
      'Old CI',
]

drugData.target = []

def drugTarget(number):
    if number > 100.0:
        return 1
    return 0

for pre in targetData:
    # choose the target
    tt = drugTarget(pre)
    drugData.target.append(tt)



drugData.target_names = [

    'States > 100k alcoholics',
    'States < 100k alcoholics',

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
drugScaled = DataFrame()

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

setupScales(drugData.data)
drugScaled.data = scaleGrid(drugData.data)
drugScaled.feature_names = drugData.feature_names
drugScaled.target = drugData.target
drugScaled.target_names = drugData.target_names

Examples = {
    'drugDefault': {
        'frame': drugData,
    },
    'drugSGD': {
        'frame': drugData,
        'mlpc': mlpc
    },
    'drugScaled': {
        'frame': drugScaled,
    },
}