import pytest
from learning import parse_csv, weighted_mode, weighted_replicate, DataSet, PluralityLearner, NaiveBayesLearner, NearestNeighborLearner
from utils import DataFile


def test_parse_csv():
    Iris = DataFile('iris.csv').read()
    assert parse_csv(Iris)[0] == [5.1,3.5,1.4,0.2,'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']

def test_Learners():
    zoo = DataSet(name="zoo")

    PluralityLearner(zoo)
    NaiveBayesLearner(zoo)
    NearestNeighborLearner(zoo,k=3)
