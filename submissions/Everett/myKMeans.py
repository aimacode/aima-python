from sklearn import datasets
from sklearn.cluster import KMeans
boston = datasets.load_boston()

Examples = {
    'BostonHouss': {
        'frame': boston,
    },
}