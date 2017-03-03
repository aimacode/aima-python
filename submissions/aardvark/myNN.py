from sklearn import datasets
from sklearn.neural_network import MLPClassifier
iris = datasets.load_iris()

Examples = {
    'IrisDefault': {
        "frame": iris,
    },
}