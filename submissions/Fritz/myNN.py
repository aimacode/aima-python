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

class DataFrame2:
    data2 = []
    feature_names2 = []
    target2 = []
    target_names2 = []

honortarget2 = []
honordata2 = DataFrame2()
honordata2.data = []


medalofhonor = medal_of_honor.get_awardees(test=True)
for issued in medalofhonor:
    try:
        date = int(issued['birth']["date"]["year"])
        honortarget.append(date)

        date2 = int(issued['awarded']["date"]["month"])
        honortarget2.append(date2)


        day = int(issued['awarded']['date']['day'])
        month = int(issued['awarded']['date']['month'])
        year = int(issued['awarded']['date']['year'])

        dayBorn = int(issued['birth']['date']['day'])
        monthBorn = int(issued['birth']['date']['month'])
        yearBorn = int(issued['birth']['date']['year'])


        honordata.data.append([day, month, year])
        honordata2.data.append([dayBorn, monthBorn, yearBorn])


    except:
        traceback.print_exc()

honordata.feature_names = [
    'day',
    'month',
    'year',
]

honordata2.feature_names = [
    'dayBorn',
    'monthBorn',
    'yearBorn',
]


honordata.target = []
honordata2.target = []



def targetdata(HDate):
    if (HDate > 1880 and HDate != -1):
        return 1
    return 0


def targetdata2(HDate2):
    if (HDate2 > 10 and HDate2 != -1):
        return 1
    return 0



for issued in honortarget:

    TD = targetdata(issued)
    honordata.target.append(TD)

honordata.target_names = [
    'Born before 1880',
    'Born after 1880',
]

for issued2 in honortarget2:

    TD2 = targetdata2(issued2)
    honordata2.target.append(TD2)

honordata2.target_names = [
    'Awarded on or before October',
    'Awarded after October',
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
Scaling the data.
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

dateScaled2 = DataFrame2()

def setupScales2(grid):
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

def scaleGrid2(grid):
    newGrid = []
    for row in range(len(grid)):
        newRow = []
        for col in range(len(grid[row])):
            try:
                cell = grid[row][col]
                scaled2 = (cell - min[col]) \
                         / (max[col] - min[col])
                newRow.append(scaled2)
            except:
                pass
        newGrid.append(newRow)
    return newGrid

setupScales(honordata2.data)
dateScaled2.data = scaleGrid2(honordata2.data)
dateScaled2.feature_names = honordata2.feature_names
dateScaled2.target = honordata2.target
dateScaled2.target_names = honordata2.target_names

Examples = {
    'Default Date':{
    'frame': honordata,
    },
    'DateSGD': {
        'frame': honordata,
        'mlpc': mlpc
    },
    'dateScaled2': {
        'frame': dateScaled,
    },
}
