from sklearn.cluster import KMeans
import traceback
from submissions.VanderKallen import slavery

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

slaveTot = DataFrame()
slaveAS = DataFrame()
slaveBoth = DataFrame()
slaveBoth2 = DataFrame()
transactions = slavery.get_transaction()
slaveTot.data = []
slaveAS.data = []
slaveBoth.data = []
slaveBoth2.data = []

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
    slaveBoth2.data.append([
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
slaveBoth2.feature_names = [
    'Children',
    'Adults',
    'Age',
    'Gender',
]

slaveTot.target = []
slaveAS.target = []
slaveBoth.target = []
slaveBoth2.target = []

def priceTarget(price):
    if price < 800:
        return 1
    return 0

def priceTarget2(price):
    if price < 400:
        return 1
    return 0

for deal in transactions:
    # choose the target
    tt = priceTarget(deal['Transaction']['Sale Details']['Price'])
    p2 = priceTarget(deal['Transaction']['Sale Details']['Price'])
    slaveTot.target.append(tt)
    slaveAS.target.append(tt)
    slaveBoth.target.append(tt)
    slaveBoth2.target.append(p2)

slaveTot.target_names = [
    'Price <= $800',
    'Price >  $800',
]

slaveAS.target_names = [
    'Price <= $800',
    'Price >  $800',
]

slaveBoth.target_names = [
    'Price <= $800',
    'Price >  $800',
]
slaveBoth2.target_names = [
    'Price <= $400',
    'Price >  $400',
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

'''
Make a customn classifier,
'''
km = KMeans(
    #n_clusters=2,
    #n_init=10,
    # init='k-means++',
    # algorithm='auto',
    # precompute_distances='auto',
    # tol=1e-4,
    # n_jobs=-1,
    # random_state=numpy.RandomState,
    # verbose=0,
    # copy_x=True,
)
km2 = KMeans(
    n_clusters=2,
    max_iter=300,
    n_init=10,
    # init='k-means++',
    # algorithm='auto',
    # precompute_distances='auto',
     tol=1e-4,
     n_jobs= -1,
     random_state= 7,
    # verbose= 0,
    # copy_x=True,
)
Examples = {
    'Regular km': {
        'frame': slaveBoth,
        'kmeans': km
    },

    'Changed km': {
        'frame': slaveBoth,
        'kmeans': km2
    },

    'Scaled Data': {
        'frame': dScaled,
    },

    'Changed target': {
        'frame': slaveBoth2,
        'kmeans': km2,
    },
}