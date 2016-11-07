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
