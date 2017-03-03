from sklearn.cluster import KMeans
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

country_classifications = []

for record in list_of_report:
    try:
        deaths = record['Data']["AIDS-Related Deaths"]["Adults"],
        target_data.append(deaths)

        country = record['Country']
        c_index = -1
        if not country in country_classifications:
            country_classifications.append(country)

        c_index = country_classifications.index(country)

        # prevalence =  int(record['Data']["HIV Prevalence"]["Adults"])
        # living =  int(record['Data']["People Living with HIV"]["Adults"])
        # new = int(record['Data']["New HIV Infections"]["Adults"])
        # deaths = int(record['Data']["AIDS-Related Deaths"]["Adults"])

        # aidsECHP.data.append([prevalence, living, new, deaths])
        aidsECHP.data.append([
            int(record['Year']),
            c_index, # Country

            int(record['Data']["HIV Prevalence"]["Adults"]),
            int(record['Data']["HIV Prevalence"]["Young Men"]),
            int(record['Data']["HIV Prevalence"]["Young Women"]),

            int(record['Data']["People Living with HIV"]["Male Adults"]),
            int(record['Data']["People Living with HIV"]["Total"]),
            int(record['Data']["People Living with HIV"]["Children"]),
            int(record['Data']["People Living with HIV"]["Female Adults"]),
            int(record['Data']["People Living with HIV"]["Adults"]),

            int(record['Data']["New HIV Infections"]["Incidence Rate Among Adults"]),
            int(record['Data']["New HIV Infections"]["Male Adults"]),
            int(record['Data']["New HIV Infections"]["All Ages"]),
            int(record['Data']["New HIV Infections"]["Children"]),
            int(record['Data']["New HIV Infections"]["Female Adults"]),
            int(record['Data']["New HIV Infections"]["Male Adults"]),

            int(record['Data']["AIDS-Related Deaths"]["Male Adults"]),
            int(record['Data']["AIDS-Related Deaths"]["All Ages"]),
            int(record['Data']["AIDS-Related Deaths"]["AIDS Orphans"]),
            int(record['Data']["AIDS-Related Deaths"]["Children"]),
            int(record['Data']["AIDS-Related Deaths"]["Female Adults"]),
            # int(record['Data']["AIDS-Related Deaths"]["Adults"]),
        ])

    except:
        traceback.print_exc()

aidsECHP.feature_names = [
    'HIV prevalence'
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

def aidsTarget(deaths):
    print(deaths)
    if deaths[0] > 20000:
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


'''
Make a customn classifier,
'''
km = KMeans(
    n_clusters=2,
    max_iter=300,
    n_init=60,
    # init='k-means++',
    init='random',
    algorithm='auto',
    # precompute_distances='auto',
    tol=1e-3,
    # n_jobs=-1,
    # random_state=numpy.RandomState,
    # verbose=0,
    # copy_x=True,
)

Examples = {
    'AidsDefault': {
        'frame': aidsECHP,
    },
    'AidsScaled': {
        'frame': aidsScaled,
    },
    'AidsCustom': {
        'frame': aidsScaled,
        'kmeans': km,
    },
}
