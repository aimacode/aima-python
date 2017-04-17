import pytest
from grid import *  # noqa


def compare_list(x, y):
    return all([elm_x == y[i] for i, elm_x in enumerate(x)])


def test_distance():
    assert distance((1, 2), (5, 5)) == 5.0


def test_distance_squared():
    assert distance_squared((1, 2), (5, 5)) == 25.0


def test_vector_clip():
    assert vector_clip((-1, 10), (0, 0), (9, 9)) == (0, 9)


def test_turn_heading():
	assert turn_heading((0, 1), 1) == (-1, 0)
	assert turn_heading((0, 1), -1) == (1, 0)
	assert turn_heading((1, 0), 1) == (0, 1)
	assert turn_heading((1, 0), -1) == (0, -1)
	assert turn_heading((0, -1), 1) == (1, 0)
	assert turn_heading((0, -1), -1) == (-1, 0)
	assert turn_heading((-1, 0), 1) == (0, -1)
	assert turn_heading((-1, 0), -1) == (0, 1)


def test_turn_left():
	assert turn_left((0, 1)) == (-1, 0)


def test_turn_right():
	assert turn_right((0, 1)) == (1, 0)


if __name__ == '__main__':
    pytest.main()
