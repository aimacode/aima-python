import pytest
from csp import *   #noqa


def test_csp_assign():
    var = 10
    val = 5
    assignment = {}
    australia.assign(var, val, assignment)

    assert australia.nassigns == 1
    assert assignment[var] == val


def test_csp_unassign():
    var = 10
    assignment = {var: 5}
    australia.unassign(var, assignment)

    assert (var in assignment) == False


def test_csp_nconflits():
    map_coloring_test = MapColoringCSP(list('RGB'), 'A: B C; B: C; C: ')
    assignment = {'A': 'R', 'B': 'G'}
    var = 'C'
    val = 'R'
    assert map_coloring_test.nconflicts(var, val, assignment) == 1

    val = 'B'
    assert map_coloring_test.nconflicts(var, val, assignment) == 0


def test_csp_actions():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    state = {'A': '1', 'B': '2', 'C': '3'}
    assert map_coloring_test.actions(state) == []

    state = {'A': '1', 'B': '3'}
    assert map_coloring_test.actions(state) == [('C', '2')]

    state = {'A': '1', 'C': '2'}
    assert map_coloring_test.actions(state) == [('B', '3')]

    state = {'A': '1'}
    assert (map_coloring_test.actions(state) == [('C', '2'), ('C', '3')] or
            map_coloring_test.actions(state) == [('B', '2'), ('B', '3')])


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
