import pytest
import math
from learning import (
    parse_csv, weighted_mode, weighted_replicate,
    rms_error, manhattan_distance, mean_boolean_error, mean_error,
    information_content
)
from utils import rounder
import statistics


def test_parse_csv():
    assert parse_csv('1, 2, 3 \n 0, 2, na') == [[1, 2, 3], [0, 2, 'na']]


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']

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
    assert statistics.mean([1, 1]) == 1
    assert 1 + 1 == 2
    assert True + True == 2
    assert statistics.mean([True, True]) == 1
    t_1 = mean_boolean_error([1,1], [0,0])
    t_2 = mean_boolean_error([0,1], [1,0])
    t_3 = mean_boolean_error([1,1], [0,1])
    t_4 = mean_boolean_error([0,0], [0,0])
    t_5 = mean_boolean_error([1,1], [1,1])
    assert [t_1, t_2, t_3, t_4, t_5] == [1, 1, 1, 1, 1]

def test_mean_error():
    assert mean_error([2,2], [2,2]) == 0
    assert mean_error([0,0], [0,1]) == 0.5
    assert mean_error([1,0], [0,1]) ==  1
    assert mean_error([0,0], [0,-1]) ==  0.5
    assert mean_error([0,0.5], [0,-0.5]) == 0.5

def test_information_content():
    # Entropy of a 4 sided fair-die
    assert information_content([1,1,1,1]) == 2
    # Entropy of a fair coin
    assert information_content([1,1]) == 1
    # Entropy of a biased coin with heads probability 0.99
    assert rounder(information_content([0.99,0.01])) == 0.0808

if __name__ == '__main__':
    pytest.main() 
