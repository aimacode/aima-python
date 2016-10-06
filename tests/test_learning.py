import pytest
from learning import parse_csv, weighted_mode, weighted_replicate


def test_parse_csv():
    assert parse_csv('1, 2, 3 \n 0, 2, na') == [[1, 2, 3], [0, 2, 'na']]


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']
