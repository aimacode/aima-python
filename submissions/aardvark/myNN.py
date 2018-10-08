from sklearn import datasets
from sklearn.neural_network import MLPClassifier

X = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]

y = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0],
     [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
     [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
     [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(4), random_state=1)

clf.fit(X, y)

y2 = clf.predict(X)

success = y == y2

iris = datasets.load_iris()

Examples = {

    'IrisDefault': {
        "frame": iris,
    },
}
