import pytest
from mdp import *  # noqa


def test_value_iteration():
    assert value_iteration(sequential_decision_environment, .01) == {(3, 2): 1.0, (3, 1): -1.0,
                                        (3, 0): 0.12958868267972745, (0, 1): 0.39810203830605462,
                                        (0, 2): 0.50928545646220924, (1, 0): 0.25348746162470537,
                                        (0, 0): 0.29543540628363629, (1, 2): 0.64958064617168676,
                                        (2, 0): 0.34461306281476806, (2, 1): 0.48643676237737926,
                                        (2, 2): 0.79536093684710951}


def test_policy_iteration():
    assert policy_iteration(sequential_decision_environment) == {(0, 0): (0, 1),  (0, 1): (0, 1), (0, 2): (1, 0),
                                            (1, 0): (1, 0),                  (1, 2): (1, 0),
                                            (2, 0): (0, 1),  (2, 1): (0, 1), (2, 2): (1, 0),
                                            (3, 0): (-1, 0), (3, 1): None,   (3, 2): None}


def test_best_policy():
    pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))
    assert sequential_decision_environment.to_arrows(pi) == [['>', '>', '>', '.'],
                                        ['^', None, '^', '.'],
                                        ['^', '>', '^', '<']]
