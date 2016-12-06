from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Ban import county_demographics

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

alumni = DataFrame()
alumni.target = []
alumni.data = []
'''
Extract data from the CORGIS elections, and merge it with the
CORGIS demographics.  Both data sets are organized by county and state.
'''

def alumniTarget(string):
    if (student['Education']["Bachelor's Degree or Higher"] > 50):
        return 1
    return 0

demographics = county_demographics.get_all_counties()
for student in demographics:
    try:
        alumni.target.append(alumniTarget(student['Education']["High School or Higher"]))

        college = float(student['Education']["High School or Higher"])
        poverty = float(student['Income']["Persons Below Poverty Level"])
        ethnicity = float(student['Ethnicities']["White Alone"])

        alumni.data.append([college, poverty, ethnicity])
    except:
        traceback.print_exc()

alumni.feature_names = [
    "High School or Higher",
    "Persons Below Poverty Level",
    "White Alone",
]

alumni.target_names = [
    'Most did not graduate college',
    'Most did graduate college',
    'Ethnicity',
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
alumniScaled = DataFrame()

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

setupScales(alumni.data)
alumniScaled.data = scaleGrid(alumni.data)
alumniScaled.feature_names = alumni.feature_names
alumniScaled.target = alumni.target
alumniScaled.target_names = alumni.target_names

Examples = {
    'AlumniDefault': {
        'Poor with degree': alumni,
    },
    'AlumniSGD': {
        'Poor with degree': alumni,
        'mlpc': mlpc
    },
    'AlumniScaled': {
        'frame': alumniScaled,
    },
}