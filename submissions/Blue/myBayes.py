import traceback

from submissions.Blue import music


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

musicATRB = DataFrame()
musicATRB.data = []
musicTarget = []
'''
Extract data from the CORGIS Music Library.

Most 'hit' songs average 48-52 bars and no more than ~3 minutes (180 seconds)...
'''

allSongs = music.get_songs()
for song in allSongs:
    try:
        length = float(song['song']["duration"])
        musicTarget.append(length)

        genre = song['artist']['terms'] #String
        title = song['song']['title'] #String
        # release = float(song['song']['Release'])

        musicATRB.data.append([genre, title])

    except:
        traceback.print_exc()

musicATRB.feature_names = [
    'Genre',
    'Title',
    'Release',
    'Length',
]

musicATRB.target = []

def musicTarget(song):
    if (song['song']['duration'] <= 240 ): #if the song is less that 4 minutes (240 seconds) long
        return 1
    return 0

# for length in musicTarget:
#     tt = musicTarget(length)
#     musicATRB.target.append(tt)

musicATRB.target_names = [
    'Not a hit song',
    'Could be a hit song',
]

Examples = {
    'Music': musicATRB,
}