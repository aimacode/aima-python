from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Ottenlips import billionaires


from submissions.Ottenlips import billionaires

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

bill = DataFrame()

list_of_billionaire = billionaires.get_billionaires()


def billtarget(billions):
    if billions<3.0:
        return 1
    else:
        return 0


for billionaires in list_of_billionaire:
    # print(billionaires['wealth']['type'])
    # print(billionaires)
    bill.target.append(billtarget(billionaires['wealth']['worth in billions']))
    # bill.target.append(billionaires['wealth']['how']['inherited'])
    bill.data.append([
        float(billionaires['demographics']['age']),
        float(billionaires['location']['gdp']),
        float(billionaires['rank']),
    ])


bill.feature_names = [
    'age',
    'gdp of origin country',
    'rank',
]

bill.target_names = [
    'very rich',
    'less rich',
]


'''
Make a customn classifier,
'''
mlpc = MLPClassifier(
            hidden_layer_sizes = (10,),
            # activation = 'relu',
            solver='sgd',
            #alpha = 0.0001,
            # batch_size='auto',
            learning_rate = 'adaptive',
            # power_t = 0.5,
            max_iter = 100, # 200,
            # shuffle = True,
            # random_state = None,
            # tol = 1e-4,
            # verbose = True,
            # warm_start = False,
            # momentum = 0.9,
            # nesterovs_momentum = True,
            # early_stopping = False,
            # validation_fraction = 0.1,
            # beta_1 = 0.9,
            # beta_2 = 0.999,
            # epsilon = 1e-8,
)
mlpcTwo = MLPClassifier(
            hidden_layer_sizes = (1000,),
            # activation = 'relu',
            solver='sgd',
            #alpha = 0.0001,
            # batch_size='auto',
            learning_rate = 'adaptive',
            # power_t = 0.5,
            max_iter = 1000, # 200,
            shuffle = True,
            # random_state = None,
            # tol = 1e-4,
            # verbose = True,
            # warm_start = False,
            # momentum = 0.9,
            # nesterovs_momentum = True,
            # early_stopping = False,
            # validation_fraction = 0.1,
            # beta_1 = 0.9,
            # beta_2 = 0.999,
            # epsilon = 1e-8,
)

billScaled = DataFrame()

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

setupScales(bill.data)
billScaled.data = scaleGrid(bill.data)
billScaled.feature_names = bill.feature_names
billScaled.target = bill.target
billScaled.target_names = bill.target_names

Examples = {

   'BillMLPC': {
        'frame': bill,
        'mlpc': mlpc,
},
 'BillMLPCTwo': {
        'frame': bill,
        'mlpc': mlpcTwo,
},
    'BillScaled':{
        'frame':billScaled,
    },
'BillScaled':{
        'frame':billScaled,
    },
    'Bill': {'frame':bill},

}

