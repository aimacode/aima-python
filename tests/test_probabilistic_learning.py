import random

import pytest

from learning import DataSet
from probabilistic_learning import *

random.seed("aima-python")


def test_naive_bayes():
    iris = DataSet(name='iris')
    # discrete
    nbd = NaiveBayesLearner(iris, continuous=False)
    assert nbd([5, 3, 1, 0.1]) == 'setosa'
    assert nbd([6, 3, 4, 1.1]) == 'versicolor'
    assert nbd([7.7, 3, 6, 2]) == 'virginica'
    # continuous
    nbc = NaiveBayesLearner(iris, continuous=True)
    assert nbc([5, 3, 1, 0.1]) == 'setosa'
    assert nbc([6, 5, 3, 1.5]) == 'versicolor'
    assert nbc([7, 3, 6.5, 2]) == 'virginica'
    # simple
    data1 = 'a' * 50 + 'b' * 30 + 'c' * 15
    dist1 = CountingProbDist(data1)
    data2 = 'a' * 30 + 'b' * 45 + 'c' * 20
    dist2 = CountingProbDist(data2)
    data3 = 'a' * 20 + 'b' * 20 + 'c' * 35
    dist3 = CountingProbDist(data3)
    dist = {('First', 0.5): dist1, ('Second', 0.3): dist2, ('Third', 0.2): dist3}
    nbs = NaiveBayesLearner(dist, simple=True)
    assert nbs('aab') == 'First'
    assert nbs(['b', 'b']) == 'Second'
    assert nbs('ccbcc') == 'Third'


if __name__ == "__main__":
    pytest.main()
