import numpy as np
import pytest

from game_theory import dominates, dominant_strategy, pure_nash_equilibria, solve_zero_sum_game

# Prisoner's dilemma, payoffs as utilities (= minus the years in prison).
# Rows/cols: 0 = testify, 1 = refuse. Outcome indexed [Ali (row)][Bo (col)].
ALI = [[-5, 0], [-10, -1]]
BO = [[-5, -10], [0, -1]]


def test_dominates():
    # for Ali, testify (row 0) strongly dominates refuse (row 1)
    assert dominates(ALI, 0, 1, strongly=True)
    assert not dominates(ALI, 1, 0, strongly=True)


def test_dominant_strategy():
    # testify (0) is the dominant strategy for both players (transpose for Bo)
    assert dominant_strategy(ALI) == 0
    assert dominant_strategy(np.transpose(BO)) == 0
    # matching pennies has no dominant strategy
    assert dominant_strategy([[1, -1], [-1, 1]]) is None


def test_pure_nash_equilibria():
    # the prisoner's dilemma has the unique (testify, testify) equilibrium
    assert pure_nash_equilibria(ALI, BO) == [(0, 0)]
    # matching pennies has no pure-strategy Nash equilibrium
    assert pure_nash_equilibria([[1, -1], [-1, 1]], [[-1, 1], [1, -1]]) == []
    # a coordination game has two pure-strategy Nash equilibria
    assert pure_nash_equilibria([[2, 0], [0, 1]], [[2, 0], [0, 1]]) == [(0, 0), (1, 1)]


def test_solve_zero_sum_game():
    # two-finger Morra: value -1/12 with optimal mixed strategy [7/12, 5/12]
    value, row_strategy, col_strategy = solve_zero_sum_game([[2, -3], [-3, 4]])
    assert value == pytest.approx(-1 / 12)
    assert row_strategy == pytest.approx([7 / 12, 5 / 12])
    assert col_strategy == pytest.approx([7 / 12, 5 / 12])

    # matching pennies: value 0, both players randomize uniformly
    value, row_strategy, col_strategy = solve_zero_sum_game([[1, -1], [-1, 1]])
    assert value == pytest.approx(0)
    assert row_strategy == pytest.approx([0.5, 0.5])
    assert col_strategy == pytest.approx([0.5, 0.5])


if __name__ == "__main__":
    pytest.main()
