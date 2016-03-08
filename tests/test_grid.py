import pytest
from aimaPy.grid import *

compare_list = lambda x, y: all([elm_x == y[i] for i, elm_x in enumerate(x)])


def test_distance():
    assert distance((1, 2), (5, 5)) == 5.0


def test_distance_squared():
    assert distance_squared((1, 2), (5, 5)) == 25.0


def test_clip():
    list_ = [clip(x, 0, 1) for x in [-1, 0.5, 10]]
    res = [0, 0.5, 1]

    assert compare_list(list_, res)


def test_vector_clip():
    assert vector_clip((-1, 10), (0, 0), (9, 9)) == (0, 9)

if __name__ == '__main__':
    pytest.main()
