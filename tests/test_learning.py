import pytest
import math
from utils import DataFile
from learning import (
    parse_csv, weighted_mode, weighted_replicate, DataSet,
    PluralityLearner, NaiveBayesLearner, NearestNeighborLearner,
    rms_error, manhattan_distance, mean_boolean_error, mean_error
)


def test_parse_csv():
    Iris = DataFile('iris.csv').read()
    assert parse_csv(Iris)[0] == [5.1, 3.5, 1.4, 0.2, 'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']


def test_plurality_learner():
    zoo = DataSet(name="zoo")

    pL = PluralityLearner(zoo)
    assert pL([]) == "mammal"


def test_naive_bayes():
    iris = DataSet(name="iris")

    nB = NaiveBayesLearner(iris)
    assert nB([5, 3, 1, 0.1]) == "setosa"


def test_k_nearest_neighbors():
    iris = DataSet(name="iris")

    kNN = NearestNeighborLearner(iris,k=3)
    assert kNN([5,3,1,0.1]) == "setosa"

def test_rms_error():
    assert rms_error([2,2], [2,2]) == 0
    assert rms_error((0,0), (0,1)) == math.sqrt(0.5)
    assert rms_error((1,0), (0,1)) ==  1
    assert rms_error((0,0), (0,-1)) ==  math.sqrt(0.5)
    assert rms_error((0,0.5), (0,-0.5)) ==  math.sqrt(0.5)

def test_manhattan_distance():
    assert manhattan_distance([2,2], [2,2]) == 0
    assert manhattan_distance([0,0], [0,1]) == 1
    assert manhattan_distance([1,0], [0,1]) ==  2
    assert manhattan_distance([0,0], [0,-1]) ==  1
    assert manhattan_distance([0,0.5], [0,-0.5]) == 1

def test_mean_boolean_error():
    assert mean_boolean_error([1,1], [0,0]) == 1
    assert mean_boolean_error([0,1], [1,0]) == 1
    assert mean_boolean_error([1,1], [0,1]) == 0.5
    assert mean_boolean_error([0,0], [0,0]) == 0
    assert mean_boolean_error([1,1], [1,1]) == 0

def test_mean_error():
    assert mean_error([2,2], [2,2]) == 0
    assert mean_error([0,0], [0,1]) == 0.5
    assert mean_error([1,0], [0,1]) ==  1
    assert mean_error([0,0], [0,-1]) ==  0.5
    assert mean_error([0,0.5], [0,-0.5]) == 0.5