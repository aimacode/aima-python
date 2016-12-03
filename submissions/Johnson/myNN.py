import traceback
from sklearn.neural_network import MLPClassifier
from submissions.Johnson import education

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

moneyvsthings = DataFrame()

joint = {}

educations = education.get_all_states()
for each in educations:
    try:
        st = each['state']
        expend = each['data']['funding']['expenditures']
        revenue = each['data']['funding']['revenue']
        ratio = each['data']['enrollment']['student teacher ratio']
        eligible = each['data']['enrollment']['students']['other']['free lunch eligible']
        grade8mathscore = each['data']['score']['math'][1]['scale score']
        enrollment = each['data']['enrollment']['students']['all']
        net = revenue - expend
        joint[st] = {}
        joint[st]['ST']= st
        joint[st]['Expend'] = expend
        joint[st]['S/T Ratio'] = ratio
        joint[st]['Net Gain'] = net
        joint[st]['Free Lunch Eligible'] = eligible
        joint[st]['Enrollment'] = enrollment
        joint[st]['8th Grade Math Score'] = grade8mathscore
    except:
        traceback.print_exc()

for st in joint:
    # choose the input values
    moneyvsthings.data.append([
        # countyST,
        # intersection[countyST]['ST'],
        # intersection[countyST]['Trump'],
        #joint[st]['Free Lunch Eligible'],
        joint[st]['S/T Ratio'],
        joint[st]['8th Grade Math Score'],
        joint[st]['Enrollment'],
        #joint[st]['Net Gain']
    ])

moneyvsthings.feature_names = [
    # 'countyST',
    # 'ST',
    # 'Trump',
    #'Free Lunch Eligible',
    'S/T Ratio',
    'Grade 8 Math Scores',
    'Enrollment'
    #'Net Gain'
]


moneyvsthings.target = []

def netpos(number):
    if number > 10000000000:
        return 1
    return 0

for st in joint:
    # choose the target
    ispos = netpos(joint[st]['Expend'])
    moneyvsthings.target.append(ispos)

moneyvsthings.target_names = [
    'Small Expenditure',
    'Large Expenditure',
    #'Free Lunch Eligible <= 300,000',
    #'Free Lunch Eligible > 300,000'
]

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

mlpc2 = MLPClassifier(
    #hidden_layer_sizes = (100,),
    # activation = 'relu',
    solver='adam', # 'adam',
     #alpha = 0.0001,
    # batch_size='auto',
    learning_rate = 'constant', # 'constant',
    # power_t = 0.5,
    max_iter = 1000, # 200,
    # shuffle = True,
    # random_state = None,
    # tol = 1e-4,
    #verbose = True,
    # warm_start = False,
    # momentum = 0.9,
    # nesterovs_momentum = True,
    # early_stopping = False,
    # validation_fraction = 0.1,
    # beta_1 = 0.9,
    # beta_2 = 0.999,
    # epsilon = 1e-8,
)

ExpendScaled = DataFrame()

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

setupScales(moneyvsthings.data)
ExpendScaled.data = scaleGrid(moneyvsthings.data)
ExpendScaled.feature_names = moneyvsthings.feature_names
ExpendScaled.target = moneyvsthings.target
ExpendScaled.target_names = moneyvsthings.target_names


Examples = {
    'ExpendDefault': {
        'frame' : moneyvsthings,
            },

    'ExpendDefault(+MLPC)': {
        'frame' : moneyvsthings,
         'mlpc': mlpc
         },

    'ExpendDefault(+MLPC2)': {
        'frame' : moneyvsthings,
         'mlpc': mlpc2
         },
    'ExpendScaled': {
        'frame' : ExpendScaled,
             },

    'ExpendScaled(+MLPC)': {
        'frame' : ExpendScaled,
         'mlpc': mlpc
             },

    'ExpendScaled(+MLPC2)': {
        'frame' : ExpendScaled,
         'mlpc': mlpc2
             }
}