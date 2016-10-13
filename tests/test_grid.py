import pytest
from grid import *  # noqa


def compare_list(x, y):
    return all([elm_x == y[i] for i, elm_x in enumerate(x)])


def test_distance():
    assert distance((1, 2), (5, 5)) == 5.0


def test_distance2():
    assert distance2((1, 2), (5, 5)) == 25.0


def test_vector_clip():
    assert vector_clip((-1, 10), (0, 0), (9, 9)) == (0, 9)

if __name__ == '__main__':
    pytest.main()
