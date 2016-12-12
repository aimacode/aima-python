import traceback
from submissions.LaMartina import state_crime
#from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

crimes = DataFrame()
larcenycr = DataFrame()

'''
Extract data from the CORGIS state_crime.
'''
joint = {}

crime = state_crime.get_all_crimes()

for c in crime:
    try:
        #if c['State'] == 'Alabama':
        stateyear = c['State'] + str(c['Year'])
        pop = c['Data']['Population']
        murder = c['Data']['Totals']['Violent']['Murder']
        assault = c['Data']['Totals']['Violent']['Assault']
        robbery = c['Data']['Totals']['Violent']['Robbery']
        rape = c['Data']['Totals']['Violent']['Rape']
        burg = c['Data']['Totals']['Property']['Burglary']
        larceny = c['Data']['Totals']['Property']['Larceny']
        motor = c['Data']['Totals']['Property']['Motor']
        joint[stateyear] = {}
        joint[stateyear]['Population'] = pop
        joint[stateyear]['Murder Numbers'] = murder
        joint[stateyear]['Rape Numbers'] = rape
        joint[stateyear]['Burglary Numbers'] = burg
        joint[stateyear]['Robbery Numbers'] = robbery
        joint[stateyear]['Assault Numbers'] = assault
        joint[stateyear]['Larceny Numbers'] = larceny
        joint[stateyear]['Motor Crime Numbers'] = motor
    except:
        traceback.print_exc()

crimes.data = []

'''
Build the input frame, row by row.
'''
for port in joint:
    # choose the input values
    crimes.data.append([
        #port,
        #joint[port]['Population'],
        joint[port]['Rape Numbers'],
        joint[port]['Robbery Numbers'],
        joint[port]['Assault Numbers']

        #joint[port]['Burglary Numbers'],
    ])
crimes.feature_names = [
    #'Population',
    'Rape Numbers',
    #'Burglary Numbers',
    'Robbery Numbers'
    'Assault Numbers'
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
crimes.target = []

def murderTarget(murdernum):
    if murdernum > 800:
        return 1
    return 0

for cri in joint:
    # choose the target
    c = murderTarget(joint[cri]['Murder Numbers'])
    crimes.target.append(c)

crimes.target_names = [
    'Murders <= 800',
    'Murders >  800',
]

'''
Try scaling the data.
'''
crimesScaled = DataFrame()

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
#The scaled data frame
setupScales(crimes.data)
crimesScaled.data = scaleGrid(crimes.data)
crimesScaled.feature_names = crimes.feature_names
crimesScaled.target = crimes.target
crimesScaled.target_names = crimes.target_names

#Creating the data frame that uses number of Larceny crimes as a target
larcenycr.data = crimes.data
larcenycr.feature_names = crimes.feature_names
larcenycr.target = []

def larcenyTarget(larcenynum):
    if larcenynum > 100000:
        return 1
    return 0

for larc in joint:
    # choose the target
    l = larcenyTarget(joint[larc]['Larceny Numbers'])
    larcenycr.target.append(l)

larcenycr.target_names = [
    'Larceny <= 100,000',
    'Murders >  100,000',
]

#Creating the scaled larceny frame
larcenyScaled = DataFrame()
setupScales(larcenycr.data)
larcenyScaled.data = scaleGrid(larcenycr.data)
larcenyScaled.feature_names = larcenycr.feature_names
larcenyScaled.target = larcenycr.target
larcenyScaled.target_names = larcenycr.target_names

#New MLPClassifier that adjusts learning rate and iterations
# mlp2 = MLPClassifier(
#     # hidden_layer_sizes = (100,),
#     # activation = 'relu',
#     #solver='sgd', # 'adam',
#     # alpha = 0.0001,
#     # batch_size='auto',
#     learning_rate = 'adaptive', # 'constant',
#     # power_t = 0.5,
#     max_iter = 1000, # 200,
#     # shuffle = True,
#     # random_state = None,
#     # tol = 1e-4,
#     # verbose = False,
#     # warm_start = False,
#     # momentum = 0.9,
#     # nesterovs_momentum = True,
#     # early_stopping = False,
#     # validation_fraction = 0.1,
#     # beta_1 = 0.9,
#     # beta_2 = 0.999,
#     # epsilon = 1e-8,
# )
# #New MLPClassifier that only adjusts the iterations
# mlp3 = MLPClassifier(
#     # hidden_layer_sizes = (100,),
#     # activation = 'relu',
#     #solver='sgd', # 'adam',
#     # alpha = 0.0001,
#     # batch_size='auto',
#     #learning_rate = 'adaptive', # 'constant',
#     # power_t = 0.5,
#     max_iter = 2000, # 200,
#     # shuffle = True,
#     # random_state = None,
#     # tol = 1e-4,
#     # verbose = False,
#     # warm_start = False,
#     # momentum = 0.9,
#     # nesterovs_momentum = True,
#     # early_stopping = False,
#     # validation_fraction = 0.1,
#     # beta_1 = 0.9,
#     # beta_2 = 0.999,
#     # epsilon = 1e-8,
# )
# #New classifier that messes with mulitple dials to get the best results
# mlp4 = MLPClassifier(
#     hidden_layer_sizes = (100,100,),
#     #activation = 'logistic',
#     #solver='lbfgs', # 'adam',
#     #alpha = 1,
#     #batch_size=1000,
#     #learning_rate = 'adaptive', # 'constant',
#     # power_t = 0.5,
#     max_iter = 1000, # 200,
#     # shuffle = True,
#     # random_state = None,
#     # tol = 1e-4,
#     # verbose = False,
#     # warm_start = False,
#     # momentum = 0.9,
#     # nesterovs_momentum = True,
#     # early_stopping = False,
#     # validation_fraction = 0.1,
#     # beta_1 = 0.9,
#     # beta_2 = 0.999,
#     # epsilon = 1e-8,
# )
# #data frame for nonviolent crimes and using them to predict murder
# nonVicrimes = DataFrame()
# nonVicrimes.data = []
#
# '''
# Build the input frame, row by row.
# '''
# for port in joint:
#     # choose the input values
#     nonVicrimes.data.append([
#         #port,
#         #joint[port]['Population'],
#         joint[port]['Burglary Numbers'],
#         joint[port]['Larceny Numbers'],
#         joint[port]['Motor Crime Numbers']
#
#         #joint[port]['Burglary Numbers'],
#     ])
# nonVicrimes.feature_names = [
#     #'Population',
#     'Burglary Numbers',
#     #'Burglary Numbers',
#     'Larceny Numbers'
#     'Motor Crime Numbers'
# ]
# nonVicrimes.target = crimes.target
# nonVicrimes.target_names = crimes.target_names
#
# #Scaled non-violent crimes data frame
# nonVicrimesScaled = DataFrame()
# setupScales(nonVicrimes.data)
# nonVicrimesScaled.data = scaleGrid(nonVicrimes.data)
# nonVicrimesScaled.feature_names = crimes.feature_names
# nonVicrimesScaled.target = crimes.target
# nonVicrimesScaled.target_names = crimes.target_names

'''
Make a customn classifier,
'''
#classifier 2
km = KMeans(
    n_clusters=6,
    # max_iter=300,
    # n_init=10,
    # init='k-means++',
    algorithm='elkan',
    # precompute_distances='auto',
    # tol=1e-4,
    # n_jobs=-1,
    # random_state=numpy.RandomState,
    # verbose=0,
    #copy_x=False,
)
#classifer 3
km2 = KMeans(
    n_clusters=15,
    # max_iter=300,
    # n_init=10,
    # init='k-means++',
    #algorithm='elkan',
    # precompute_distances='auto',
    # tol=1e-4,
    # n_jobs=-1,
    # random_state=numpy.RandomState,
    # verbose=0,
    #copy_x=False,
)

Examples = {
    'Crimes': {
        'frame': crimes,
    },
    'CrimesScaled': {
        'frame': crimesScaled,
    },
    'CrimesClassifier2': {
        'frame': crimes,
        'kmeans': km
    },
    'CrimesClassifier2Scaled': {
        'frame': crimesScaled,
        'kmeans': km
    },
    'Larceny': {
        'frame': larcenycr,
        #'kmeans': km
    },
    'LarcenyScaled':{
        'frame': larcenyScaled
    },
    'LarcenyClassifier3': {
        'frame': larcenycr,
        'kmeans': km2
    },
    'LarcenyClassifier3Scaled': {
        'frame': larcenyScaled,
        'kmeans': km2
    },
    # 'CrimesMLP4': {
    #     'frame': crimes,
    #     'mlp4': mlp4
    # },
    # # 'CrimesMLP4Scaled': {
    # #     'frame': crimesScaled,
    # #     'mlp4': mlp4
    # # },
    # 'Non-Violent Crimes': {
    #     'frame': nonVicrimes,
    # },
    # 'Non-Violent Crimes Scaled': {
    #     'frame': nonVicrimesScaled,
    # },
}
