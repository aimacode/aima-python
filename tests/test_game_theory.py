import numpy as np
import pytest

from game_theory import (dominates, dominant_strategy, pure_nash_equilibria, solve_zero_sum_game,
                         shapley_value, is_in_core, plurality_winner, borda_winner, condorcet_winner,
                         vickrey_auction)

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


def gloves(coalition):
    """Glove market: players 1 and 2 hold a left glove, player 3 a right glove;
    a coalition is worth the number of complete pairs it can make."""
    lefts = len({1, 2} & coalition)
    rights = len({3} & coalition)
    return min(lefts, rights)


def test_shapley_value():
    phi = shapley_value([1, 2, 3], gloves)
    # the scarce right glove (player 3) captures most of the value
    assert phi == pytest.approx({1: 1 / 6, 2: 1 / 6, 3: 2 / 3})
    # the Shapley value is efficient: it distributes the whole grand-coalition value
    assert sum(phi.values()) == pytest.approx(gloves(frozenset({1, 2, 3})))


def test_is_in_core():
    # giving the whole value to the indispensable player 3 is in the core
    assert is_in_core([1, 2, 3], gloves, {1: 0, 2: 0, 3: 1})
    # splitting it with player 1 lets coalition {2, 3} object (they can make a pair)
    assert not is_in_core([1, 2, 3], gloves, {1: 0.5, 2: 0, 3: 0.5})


# Condorcet's paradox (Equation 18.2): pairwise majority preference is cyclic
CONDORCET_PARADOX = [['a', 'b', 'c'], ['c', 'a', 'b'], ['b', 'c', 'a']]
# an election where plurality, Borda and Condorcet disagree
ELECTION = [['a', 'b', 'c']] * 4 + [['b', 'c', 'a']] * 3 + [['c', 'b', 'a']] * 2


def test_plurality_winner():
    assert plurality_winner(ELECTION) == 'a'  # 4 first-place votes


def test_borda_winner():
    assert borda_winner(ELECTION) == 'b'  # compromise candidate wins on points


def test_condorcet_winner():
    assert condorcet_winner(ELECTION) == 'b'  # b beats both a and c pairwise
    assert condorcet_winner(CONDORCET_PARADOX) is None  # cyclic majority: no winner


def test_vickrey_auction():
    winner, price = vickrey_auction({'a': 10, 'b': 8, 'c': 5})
    assert winner == 'a'  # highest bidder wins
    assert price == 8  # but pays the second-highest bid


if __name__ == "__main__":
    pytest.main()
