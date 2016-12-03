from sklearn import datasets

from sklearn.cluster import KMeans
# import numpy
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


def billtarget(age):
    if age<40:
        return 1
    else:
        return 0


for billionaires in list_of_billionaire:
    # print(billionaires['wealth']['type'])
    # print(billionaires)
    bill.target.append(billionaires['demographics']['age'])
    # bill.target.append(billionaires['wealth']['how']['inherited'])
    bill.data.append([
         float( billionaires['wealth']['worth in billions']),
        #float(billionaires['demographics']['age']),
          # float(billionaires['location']['gdp']),
         float(billionaires['rank']),
    ])


bill.feature_names = [
    # 'age',
    'wealth'
    # 'gdp of origin country',
    'rank',
]

bill.target_names = [
    'Young',
    'Old',
]


'''
Make a customn classifier,
'''
km = KMeans(
    n_clusters=15,
     # max_iter=3000,
    #  n_init=10,
    # init='k-means++',
    #  algorithm='auto',
    #  precompute_distances='auto',
    # tol=1e-4,
    #  n_jobs=-1,
    #  random_state=numpy.RandomState,
    #  verbose=1,
    #  copy_x=True,
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
    'Billdefault': {
        'frame': bill,
    },

   'BillKMClassifier': {
        'frame': bill,
        'kmeans': km,
},



}

