import pytest
from csp import *   #noqa


def test_backtracking_search():
    assert (backtracking_search(australia) is not None) == True
    assert (backtracking_search(australia, select_unassigned_variable=mrv) is not None) == True
    assert (backtracking_search(australia, order_domain_values=lcv) is not None) == True
    assert (backtracking_search(australia, select_unassigned_variable=mrv,
                                order_domain_values=lcv) is not None) == True
    assert (backtracking_search(australia, inference=forward_checking) is not None) == True
    assert (backtracking_search(australia, inference=mac) is not None) == True
    assert (backtracking_search(usa, select_unassigned_variable=mrv,
                                order_domain_values=lcv, inference=mac) is not None) == True


def test_universal_dict():
    d = UniversalDict(42)
    assert d['life'] == 42


def test_parse_neighbours():
    assert parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}


if __name__ == "__main__":
    pytest.main()
