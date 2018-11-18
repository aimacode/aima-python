from sklearn import datasets
from sklearn.cluster import KMeans
diabetes = datasets.load_diabetes()

Examples = {
    'Diabetus': {
        'frame': diabetes,
    },
}