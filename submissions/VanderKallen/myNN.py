import traceback
from submissions.VanderKallen import slavery
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

slaveTot = DataFrame()
slaveAS = DataFrame()
slaveBoth = DataFrame()
transactions = slavery.get_transaction()
slaveTot.data = []
slaveAS.data = []
slaveBoth.data = []

def genderNumber(gender):
    if gender == 'M':
        return 1
    else:
        return 0

for seller in transactions:
    # choose the input values
    slaveTot.data.append([
        seller['Transaction']['Number of Child Slaves'],
        seller['Transaction']['Number of Adult Slaves'],
        ])
    slaveAS.data.append([
        seller['Slave']['Age'],
        genderNumber(seller['Slave']['Gender']),
    ])
    slaveBoth.data.append([
        seller['Transaction']['Number of Child Slaves'],
        seller['Transaction']['Number of Adult Slaves'],
        seller['Slave']['Age'],
        genderNumber(seller['Slave']['Gender']),
    ])

slaveAS.feature_names = [
    'Age',
    'Gender',
]

slaveTot.feature_names = [
    'Children',
    'Adults',
]

slaveBoth.feature_names = [
    'Children',
    'Adults',
    'Age',
    'Gender',
]

slaveTot.target = []
slaveAS.target = []
slaveBoth.target = []

def priceTarget(price):
    if price < 700:
        return 1
    return 0

for deal in transactions:
    # choose the target
    tt = priceTarget(deal['Transaction']['Sale Details']['Price'])
    slaveTot.target.append(tt)
    slaveAS.target.append(tt)
    slaveBoth.target.append(tt)

slaveTot.target_names = [
    'Price <= $200',
    'Price >  $200',
]

slaveAS.target_names = [
    'Price <= $200',
    'Price >  $200',
]

slaveBoth.target_names = [
    'Price <= $200',
    'Price >  $200',
]

dScaled = DataFrame()

def setScale(grid):
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

setScale(slaveTot.data)
dScaled.data = scaleGrid(slaveTot.data)
dScaled.feature_names = slaveTot.feature_names
dScaled.target = slaveTot.target
dScaled.target_names = slaveTot.target_names

mlpc = MLPClassifier(
    solver='sgd',
    learning_rate = 'adaptive',
    alpha= .0001
)
mlpr = MLPRegressor(
    solver='sgd',
    learning_rate = 'adaptive',
    momentum= .1
)

mlpr2 = MLPRegressor(
    solver='adam',
    beta_1= .9,
    beta_2= .9
)


Examples = {
    'Scaled Data Frame' : {
        'frame': dScaled,
        'mlpr': mlpr
    },
    'Age and Sex: W/ mlpc' : {
        'frame': slaveAS,
    },

    'Total, Age, and Sex: w/ mlpr2' : {
        'frame': slaveBoth,
        'mlpr2': mlpr2
    },

    'Total, Age, and Sex: w/ mlpc' : {
        'frame': slaveBoth,
        'mlpc' : mlpc
    }
}