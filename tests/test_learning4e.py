import pytest
import math
import random
from utils import open_data
from learning import *


random.seed("aima-python")


def test_mean_boolean_error():
    assert mean_boolean_error([1, 1], [0, 0]) == 1
    assert mean_boolean_error([0, 1], [1, 0]) == 1
    assert mean_boolean_error([1, 1], [0, 1]) == 0.5
    assert mean_boolean_error([0, 0], [0, 0]) == 0
    assert mean_boolean_error([1, 1], [1, 1]) == 0


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
