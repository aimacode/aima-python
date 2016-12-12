from sklearn.cluster import KMeans
import traceback
from submissions.Blue import music


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

musicATRB = DataFrame()
musicATRB.data = []
targetData = []

'''
Extract data from the CORGIS Music Library.

Most 'hit' songs average 48-52 bars and no more than ~3 minutes (180 seconds)...
'''

allSongs = music.get_songs()
for song in allSongs:
    try:
        length = float(song['song']["duration"])
        targetData.append(length)

        genre = song['artist']['terms'] #String
        title = song['song']['title'] #String
        length = float(song['song']['duration'])
        year = float(song['song']['year'])
        catchy = float(song['song']['familiarity'])
        musicATRB.data.append([genre, title])

    except:
        traceback.print_exc()

musicATRB.feature_names = [
    'Genre',
    'Title',
    'Year',
    'Length',
    'Familiarity'
]

musicATRB.target = []

def musicTarget(length):
    if (length <= 210 and ("pop" in genre) ): #if the song is less that 3.5 minutes (210 seconds) long and if has 'pop' its description
        return 1
    return 0


for i in targetData:
    tt = musicTarget(i)
    musicATRB.target.append(tt)

musicATRB.target_names = [
    'Not a hit song',
    'Could be a hit song',
]

Examples = {
    'Music': musicATRB,
}

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


setupScales(musicATRB.data)
musicScaled.data = scaleGrid(musicATRB.data)
musicScaled.feature_names = musicATRB.feature_names
musicScaled.target = musicATRB.target
musicScaled.target_names = musicATRB.target_names
catchy.data = musicATRB.data
catchy.feature_names = musicATRB.feature_names


catchy.target = []

catchy.target_names = [
    'Catchy'
    'Not Catchy'
]


def catchyTarget(catchy):
    if (catchy >= 0.5):
        return 1
    return 0

for i in targetData:
    tt = musicTarget(i)
    catchy.target.append(tt)


catchyScaled = DataFrame()
setupScales(catchy.data)
catchyScaled.data = scaleGrid(catchy.data)
catchyScaled.feature_names = catchy.feature_names
catchyScaled.target = catchy.target
catchyScaled.target_names = catchy.target_names


'''
Make a custom classifier,
'''
km = KMeans(
    n_clusters=2,
    max_iter=300,
    # n_init=10,
    init='k-means++',
    algorithm='auto',
    # precompute_distances='auto',
    # tol=1e-4,
    n_jobs=-1,
    # random_state=numpy.RandomState,
    verbose=0,
    # copy_x=True,
)

km_1 = KMeans(
    n_clusters=4,
    max_iter=500,
    # n_init=10,
    # init='k-means++',
    algorithm='full',
    # precompute_distances='auto',
    # tol=1e-4,
    # n_jobs=-1,
    # random_state=numpy.RandomState,
    verbose=1,
    # copy_x=True,
)

Examples = {
    'musicDefault': {
        'frame': musicATRB,
    },

    'MusicScaled': {
        'frame': musicScaled,
        'kmeans' : km
    },
    'CatchyScaled': {
        'frame': catchyScaled,
        'kmeans': km
    },
'MusicScaled': {
        'frame': musicScaled,
        'kmeans' : km_1
    },
    'CatchyScaled': {
        'frame': catchyScaled,
        'kmeans': km_1
    },
}