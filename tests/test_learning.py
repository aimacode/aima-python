import pytest
import math
import random
from utils import open_data
from learning import *


random.seed("aima-python")


def test_euclidean():
    distance = euclidean_distance([1, 2], [3, 4])
    assert round(distance, 2) == 2.83

    distance = euclidean_distance([1, 2, 3], [4, 5, 6])
    assert round(distance, 2) == 5.2

    distance = euclidean_distance([0, 0, 0], [0, 0, 0])
    assert distance == 0

def test_cross_entropy():
    loss = cross_entropy_loss([1,0], [0.9, 0.3])
    assert round(loss,2) == 0.23

    loss = cross_entropy_loss([1,0,0,1], [0.9,0.3,0.5,0.75])
    assert round(loss,2) == 0.36

    loss = cross_entropy_loss([1,0,0,1,1,0,1,1], [0.9,0.3,0.5,0.75,0.85,0.14,0.93,0.79])
    assert round(loss,2) == 0.26


def test_rms_error():
    assert rms_error([2, 2], [2, 2]) == 0
    assert rms_error((0, 0), (0, 1)) == math.sqrt(0.5)
    assert rms_error((1, 0), (0, 1)) ==  1
    assert rms_error((0, 0), (0, -1)) ==  math.sqrt(0.5)
    assert rms_error((0, 0.5), (0, -0.5)) ==  math.sqrt(0.5)


def test_manhattan_distance():
    assert manhattan_distance([2, 2], [2, 2]) == 0
    assert manhattan_distance([0, 0], [0, 1]) == 1
    assert manhattan_distance([1, 0], [0, 1]) ==  2
    assert manhattan_distance([0, 0], [0, -1]) ==  1
    assert manhattan_distance([0, 0.5], [0, -0.5]) == 1


def test_mean_boolean_error():
    assert mean_boolean_error([1, 1], [0, 0]) == 1
    assert mean_boolean_error([0, 1], [1, 0]) == 1
    assert mean_boolean_error([1, 1], [0, 1]) == 0.5
    assert mean_boolean_error([0, 0], [0, 0]) == 0
    assert mean_boolean_error([1, 1], [1, 1]) == 0


def test_mean_error():
    assert mean_error([2, 2], [2, 2]) == 0
    assert mean_error([0, 0], [0, 1]) == 0.5
    assert mean_error([1, 0], [0, 1]) ==  1
    assert mean_error([0, 0], [0, -1]) ==  0.5
    assert mean_error([0, 0.5], [0, -0.5]) == 0.5


def test_exclude():
    iris = DataSet(name='iris', exclude=[3])
    assert iris.inputs == [0, 1, 2]


def test_parse_csv():
    Iris = open_data('iris.csv').read()
    assert parse_csv(Iris)[0] == [5.1, 3.5, 1.4, 0.2, 'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']


def test_means_and_deviation():
    iris = DataSet(name="iris")

    means, deviations = iris.find_means_and_deviations()

    assert round(means["setosa"][0], 3) == 5.006
    assert round(means["versicolor"][0], 3) == 5.936
    assert round(means["virginica"][0], 3) == 6.588

    assert round(deviations["setosa"][0], 3) == 0.352
    assert round(deviations["versicolor"][0], 3) == 0.516
    assert round(deviations["virginica"][0], 3) == 0.636


def test_plurality_learner():
    zoo = DataSet(name="zoo")

    pL = PluralityLearner(zoo)
    assert pL([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1]) == "mammal"


def test_naive_bayes():
    iris = DataSet(name="iris")

    # Discrete
    nBD = NaiveBayesLearner(iris, continuous=False)
    assert nBD([5, 3, 1, 0.1]) == "setosa"
    assert nBD([6, 3, 4, 1.1]) == "versicolor"
    assert nBD([7.7, 3, 6, 2]) == "virginica"

    # Continuous
    nBC = NaiveBayesLearner(iris, continuous=True)
    assert nBC([5, 3, 1, 0.1]) == "setosa"
    assert nBC([6, 5, 3, 1.5]) == "versicolor"
    assert nBC([7, 3, 6.5, 2]) == "virginica"

    # Simple
    data1 = 'a'*50 + 'b'*30 + 'c'*15
    dist1 = CountingProbDist(data1)
    data2 = 'a'*30 + 'b'*45 + 'c'*20
    dist2 = CountingProbDist(data2)
    data3 = 'a'*20 + 'b'*20 + 'c'*35
    dist3 = CountingProbDist(data3)

    dist = {('First', 0.5): dist1, ('Second', 0.3): dist2, ('Third', 0.2): dist3}
    nBS = NaiveBayesLearner(dist, simple=True)
    assert nBS('aab') == 'First'
    assert nBS(['b', 'b']) == 'Second'
    assert nBS('ccbcc') == 'Third'


def test_k_nearest_neighbors():
    iris = DataSet(name="iris")
    kNN = NearestNeighborLearner(iris, k=3)
    assert kNN([5, 3, 1, 0.1]) == "setosa"
    assert kNN([5, 3, 1, 0.1]) == "setosa"
    assert kNN([6, 5, 3, 1.5]) == "versicolor"
    assert kNN([7.5, 4, 6, 2]) == "virginica"


def test_truncated_svd():
    test_mat = [[17, 0],
                [0, 11]]
    _, _, eival = truncated_svd(test_mat)
    assert isclose(abs(eival[0]), 17)
    assert isclose(abs(eival[1]), 11)

    test_mat = [[17, 0],
                [0, -34]]
    _, _, eival = truncated_svd(test_mat)
    assert isclose(abs(eival[0]), 34)
    assert isclose(abs(eival[1]), 17)

    test_mat = [[1, 0, 0, 0, 2],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0]]
    _, _, eival = truncated_svd(test_mat)
    assert isclose(abs(eival[0]), 3)
    assert isclose(abs(eival[1]), 5**0.5)

    test_mat = [[3, 2, 2],
                [2, 3, -2]]
    _, _, eival = truncated_svd(test_mat)
    assert isclose(abs(eival[0]), 5)
    assert isclose(abs(eival[1]), 3)


def test_decision_tree_learner():
    iris = DataSet(name="iris")
    dTL = DecisionTreeLearner(iris)
    assert dTL([5, 3, 1, 0.1]) == "setosa"
    assert dTL([6, 5, 3, 1.5]) == "versicolor"
    assert dTL([7.5, 4, 6, 2]) == "virginica"


def test_information_content():
    assert information_content([]) == 0
    assert information_content([4]) == 0
    assert information_content([5, 4, 0, 2, 5, 0]) > 1.9
    assert information_content([5, 4, 0, 2, 5, 0]) < 2
    assert information_content([1.5, 2.5]) > 0.9
    assert information_content([1.5, 2.5]) < 1.0


def test_random_forest():
    iris = DataSet(name="iris")
    rF = RandomForest(iris)
    tests = [([5.0, 3.0, 1.0, 0.1], "setosa"),
             ([5.1, 3.3, 1.1, 0.1], "setosa"),
             ([6.0, 5.0, 3.0, 1.0], "versicolor"),
             ([6.1, 2.2, 3.5, 1.0], "versicolor"),
             ([7.5, 4.1, 6.2, 2.3], "virginica"),
             ([7.3, 3.7, 6.1, 2.5], "virginica")]
    assert grade_learner(rF, tests) >= 1/3


def test_neural_network_learner():
    iris = DataSet(name="iris")
    classes = ["setosa", "versicolor", "virginica"]
    iris.classes_to_numbers(classes)
    nNL = NeuralNetLearner(iris, [5], 0.15, 75)
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    assert grade_learner(nNL, tests) >= 1/3
    assert err_ratio(nNL, iris) < 0.21


def test_perceptron():
    iris = DataSet(name="iris")
    iris.classes_to_numbers()
    classes_number = len(iris.values[iris.target])
    perceptron = PerceptronLearner(iris)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(perceptron, tests) > 1/2
    assert err_ratio(perceptron, iris) < 0.4


def test_random_weights():
    min_value = -0.5
    max_value = 0.5
    num_weights = 10
    test_weights = random_weights(min_value, max_value, num_weights)
    assert len(test_weights) == num_weights
    for weight in test_weights:
        assert weight >= min_value and weight <= max_value


def test_adaboost():
    iris = DataSet(name="iris")
    iris.classes_to_numbers()
    WeightedPerceptron = WeightedLearner(PerceptronLearner)
    AdaboostLearner = AdaBoost(WeightedPerceptron, 5)
    adaboost = AdaboostLearner(iris)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(adaboost, tests) > 4/6
    assert err_ratio(adaboost, iris) < 0.25
