import pytest

from learning import *

random.seed("aima-python")


def test_exclude():
    iris = DataSet(name='iris', exclude=[3])
    assert iris.inputs == [0, 1, 2]


def test_parse_csv():
    iris = open_data('iris.csv').read()
    assert parse_csv(iris)[0] == [5.1, 3.5, 1.4, 0.2, 'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']


def test_means_and_deviation():
    iris = DataSet(name='iris')
    means, deviations = iris.find_means_and_deviations()
    assert round(means['setosa'][0], 3) == 5.006
    assert round(means['versicolor'][0], 3) == 5.936
    assert round(means['virginica'][0], 3) == 6.588
    assert round(deviations['setosa'][0], 3) == 0.352
    assert round(deviations['versicolor'][0], 3) == 0.516
    assert round(deviations['virginica'][0], 3) == 0.636


def test_plurality_learner():
    zoo = DataSet(name='zoo')
    pl = PluralityLearner(zoo)
    assert pl([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1]) == 'mammal'


def test_k_nearest_neighbors():
    iris = DataSet(name='iris')
    knn = NearestNeighborLearner(iris, k=3)
    assert knn([5, 3, 1, 0.1]) == 'setosa'
    assert knn([6, 5, 3, 1.5]) == 'versicolor'
    assert knn([7.5, 4, 6, 2]) == 'virginica'


def test_decision_tree_learner():
    iris = DataSet(name='iris')
    dtl = DecisionTreeLearner(iris)
    assert dtl([5, 3, 1, 0.1]) == 'setosa'
    assert dtl([6, 5, 3, 1.5]) == 'versicolor'
    assert dtl([7.5, 4, 6, 2]) == 'virginica'


def test_svc():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = (np.array([x[:n_features] for x in iris.examples]),
            np.array([x[n_features] for x in iris.examples]))
    svm = MultiClassLearner(SVC()).fit(X, y)
    assert svm.predict([[5.0, 3.1, 0.9, 0.1]]) == 0
    assert svm.predict([[5.1, 3.5, 1.0, 0.0]]) == 0
    assert svm.predict([[4.9, 3.3, 1.1, 0.1]]) == 0
    assert svm.predict([[6.0, 3.0, 4.0, 1.1]]) == 1
    assert svm.predict([[6.1, 2.2, 3.5, 1.0]]) == 1
    assert svm.predict([[5.9, 2.5, 3.3, 1.1]]) == 1
    assert svm.predict([[7.5, 4.1, 6.2, 2.3]]) == 2
    assert svm.predict([[7.3, 4.0, 6.1, 2.4]]) == 2
    assert svm.predict([[7.0, 3.3, 6.1, 2.5]]) == 2


def test_information_content():
    assert information_content([]) == 0
    assert information_content([4]) == 0
    assert information_content([5, 4, 0, 2, 5, 0]) > 1.9
    assert information_content([5, 4, 0, 2, 5, 0]) < 2
    assert information_content([1.5, 2.5]) > 0.9
    assert information_content([1.5, 2.5]) < 1.0


def test_random_forest():
    iris = DataSet(name='iris')
    rf = RandomForest(iris)
    tests = [([5.0, 3.0, 1.0, 0.1], 'setosa'),
             ([5.1, 3.3, 1.1, 0.1], 'setosa'),
             ([6.0, 5.0, 3.0, 1.0], 'versicolor'),
             ([6.1, 2.2, 3.5, 1.0], 'versicolor'),
             ([7.5, 4.1, 6.2, 2.3], 'virginica'),
             ([7.3, 3.7, 6.1, 2.5], 'virginica')]
    assert grade_learner(rf, tests) >= 1 / 3


def test_neural_network_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    nnl = NeuralNetLearner(iris, [5], 0.15, 75)
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    assert grade_learner(nnl, tests) >= 1 / 3
    assert err_ratio(nnl, iris) < 0.21


def test_perceptron():
    iris = DataSet(name='iris')
    iris.classes_to_numbers()
    pl = PerceptronLearner(iris)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(pl, tests) > 1 / 2
    assert err_ratio(pl, iris) < 0.4


def test_random_weights():
    min_value = -0.5
    max_value = 0.5
    num_weights = 10
    test_weights = random_weights(min_value, max_value, num_weights)
    assert len(test_weights) == num_weights
    for weight in test_weights:
        assert min_value <= weight <= max_value


def test_ada_boost():
    iris = DataSet(name='iris')
    iris.classes_to_numbers()
    wl = WeightedLearner(PerceptronLearner)
    ab = ada_boost(iris, wl, 5)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(ab, tests) > 2 / 3
    assert err_ratio(ab, iris) < 0.25


if __name__ == "__main__":
    pytest.main()
