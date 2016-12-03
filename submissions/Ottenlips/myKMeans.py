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

def billtarget(num):

    if num>200:
        return 0

    if num>500:
        return 1
    else:
        return 2


for billionaires in list_of_billionaire:
    # print(billionaires['wealth']['type'])
    # print(billionaires)
    bill.target.append(billtarget(float(billionaires['rank'])))
    # bill.target.append(billionaires['wealth']['how']['inherited'])
    bill.data.append([
        billionaires['wealth']['worth in billions'],
        float(billionaires['demographics']['age']),
        float(billionaires['location']['gdp']),

    ])



bill.feature_names = [
    # 'age',
   'wealth',
'age',
   'gdp of origin country',

    # 'gdp of origin country',
    # 'rank',
]

bill.target_names = [

    'high rank',
    'med rank',
    'lowest rank',

]


'''
Make a customn classifier,
'''
km = KMeans(
    n_clusters=12,
       # max_iter=1,
     # n_init=12,
     #   init='',
    #  algorithm='auto',
       precompute_distances='auto',
    #  tol=1e-4,
    #  n_jobs=-1,
    #  random_state=numpy.RandomState,
    #  verbose=1,
    #  copy_x=True,
)
km2 = KMeans(
    n_clusters=2,
     # max_iter=20,
      n_init=50,
       init='random',
      algorithm='auto',
    #    precompute_distances='auto',
    #  tol=1e-4,
    #   n_jobs=-1,
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
 'BillKMClassifierScaled': {
        'frame': billScaled,
        'kmeans': km2,
},


}

