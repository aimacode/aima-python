from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Conklin import music

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

musicECHP = DataFrame()
musicECHP.data = []
targetInfo = []

list_of_songs = music.get_songs()
for song in list_of_songs:
    try:
        tempo = float(song['song']["tempo"])
        targetInfo.append(tempo)

        loudness = float(song['song']["loudness"])
        fadeOut = float(song['song']["start_of_fade_out"])
        fadeIn = float(song['song']["end_of_fade_in"])
        duration = float(song['song']["duration"])
        releaseYear = float(song['song']["year"])

        musicECHP.data.append([tempo, fadeOut, fadeIn, duration, releaseYear])

    except:
        traceback.print_exc()


musicECHP.feature_names = [
    'Loudness',
    'Fade Out',
    'Fade In',
    'Duration',
    'Release Year'
]

musicECHP.target = []

def musicTarget(speed):
    if speed > 100:
        return 1
    return 0

for pre in targetInfo:
    # choose the target
    tt = musicTarget(pre)
    musicECHP.target.append(tt)

musicECHP.target_names = [
    'Tempo <= 100 bpm',
    'Tempo > 100 bpm',
]

Examples = {
    'Music': musicECHP,
}

'''
Make a custom classifier,
'''
mlpc = MLPClassifier(
    hidden_layer_sizes = (1000,),
    activation = 'relu',
    solver='sgd',#'adam',
    # alpha = 0.0001,
    # batch_size='auto',
    learning_rate = 'adaptive', # 'constant',
    # power_t = 0.5,
    max_iter = 1000, # 200,
    shuffle = False,
    # random_state = None,
    # tol = 1e-4,
    # verbose = False,
    # warm_start = False,
     momentum = 0.5,
    # nesterovs_momentum = True,
    # early_stopping = False,
    # validation_fraction = 0.1,
    beta_1 = 0.9,
    beta_2 = 0.999,
    # epsilon = 1e-8,
)

'''
Try scaling the data.
'''
musicScaled = DataFrame()

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

setupScales(musicECHP.data)
musicScaled.data = scaleGrid(musicECHP.data)
musicScaled.feature_names = musicECHP.feature_names
musicScaled.target = musicECHP.target
musicScaled.target_names = musicECHP.target_names

Examples = {
    'MusicDefault': {
        'frame': musicECHP,
    },
    'MusicSGD': {
        'frame': musicECHP,
        'mlpc': mlpc
    },
    'MusicScaled': {
        'frame': musicScaled,
    },
}
