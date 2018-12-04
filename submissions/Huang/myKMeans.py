from sklearn import datasets
from submissions.Huang import music
from sklearn.cluster import KMeans
iris = datasets.load_iris()

Examples = {
    'IrisDefault': {
        'frame': iris,
    },
    'Music': {
        'frame': music,
    }
}