import pytest
from csp import *   # noqa


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

    assert var not in assignment


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

    state = (('A', '1'), ('B', '3'))
    assert map_coloring_test.actions(state) == [('C', '2')]

    state = {'A': '1'}
    assert (map_coloring_test.actions(state) == [('C', '2'), ('C', '3')] or
            map_coloring_test.actions(state) == [('B', '2'), ('B', '3')])


def test_csp_result():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    state = (('A', '1'), ('B', '3'))
    action = ('C', '2')

    assert map_coloring_test.result(state, action) == (('A', '1'), ('B', '3'), ('C', '2'))


def test_csp_goal_test():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    state = (('A', '1'), ('B', '3'), ('C', '2'))
    assert map_coloring_test.goal_test(state) is True

    state = (('A', '1'), ('C', '2'))
    assert map_coloring_test.goal_test(state) is False


def test_csp_support_pruning():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.support_pruning()
    assert map_coloring_test.curr_domains == {'A': ['1', '2', '3'], 'B': ['1', '2', '3'],
                                              'C': ['1', '2', '3']}


def test_csp_suppose():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    var = 'A'
    value = '1'

    removals = map_coloring_test.suppose(var, value)

    assert removals == [('A', '2'), ('A', '3')]
    assert map_coloring_test.curr_domains == {'A': ['1'], 'B': ['1', '2', '3'],
                                              'C': ['1', '2', '3']}


def test_csp_prune():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    removals = None
    var = 'A'
    value = '3'

    map_coloring_test.support_pruning()
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.curr_domains == {'A': ['1', '2'], 'B': ['1', '2', '3'],
                                              'C': ['1', '2', '3']}
    assert removals is None

    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    removals = [('A', '2')]
    map_coloring_test.support_pruning()
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.curr_domains == {'A': ['1', '2'], 'B': ['1', '2', '3'],
                                              'C': ['1', '2', '3']}
    assert removals == [('A', '2'), ('A', '3')]


def test_csp_choices():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    var = 'A'
    assert map_coloring_test.choices(var) == ['1', '2', '3']

    map_coloring_test.support_pruning()
    removals = None
    value = '3'
    map_coloring_test.prune(var, value, removals)
    assert map_coloring_test.choices(var) == ['1', '2']


def test_csp_infer_assignement():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.infer_assignment() == {}

    var = 'A'
    value = '3'
    map_coloring_test.prune(var, value, None)
    value = '1'
    map_coloring_test.prune(var, value, None)

    map_coloring_test.infer_assignment() == {'A': '2'}


def test_csp_restore():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.curr_domains = {'A': ['2', '3'], 'B': ['1'], 'C': ['2', '3']}
    removals = [('A', '1'), ('B', '2'), ('B', '3')]

    map_coloring_test.restore(removals)

    assert map_coloring_test.curr_domains == {'A': ['2', '3', '1'], 'B': ['1', '2', '3'],
                                              'C': ['2', '3']}


def test_csp_conflicted_vars():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')

    current = {}
    var = 'A'
    val = '1'
    map_coloring_test.assign(var, val, current)

    var = 'B'
    val = '3'
    map_coloring_test.assign(var, val, current)

    var = 'C'
    val = '3'
    map_coloring_test.assign(var, val, current)

    conflicted_vars = map_coloring_test.conflicted_vars(current)

    assert (conflicted_vars == ['B', 'C'] or conflicted_vars == ['C', 'B'])


def test_revise():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0], 'B': [4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x+y) == 4

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()
    Xi = 'A'
    Xj = 'B'
    removals = []

    assert revise(csp, Xi, Xj, removals) is False
    assert len(removals) == 0

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assert revise(csp, Xi, Xj, removals) is True
    assert removals == [('A', 1), ('A', 3)]


def test_AC3():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x+y) == 4 and y % 2 != 0
    removals = []

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3(csp, removals=removals) is False

    constraints = lambda X, x, Y, y: (x % 2) == 0 and (x+y) == 4
    removals = []
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3(csp, removals=removals) is True
    assert (removals == [('A', 1), ('A', 3), ('B', 1), ('B', 3)] or
            removals == [('B', 1), ('B', 3), ('A', 1), ('A', 3)])


def test_first_unassigned_variable():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    assignment = {'A': '1', 'B': '2'}
    assert first_unassigned_variable(assignment, map_coloring_test) == 'C'

    assignment = {'B': '1'}
    assert (first_unassigned_variable(assignment, map_coloring_test) == 'A' or
            first_unassigned_variable(assignment, map_coloring_test) == 'C')


def test_num_legal_values():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    map_coloring_test.support_pruning()
    var = 'A'
    assignment = {}

    assert num_legal_values(map_coloring_test, var, assignment) == 3

    map_coloring_test = MapColoringCSP(list('RGB'), 'A: B C; B: C; C: ')
    assignment = {'A': 'R', 'B': 'G'}
    var = 'C'

    assert num_legal_values(map_coloring_test, var, assignment) == 1


def test_mrv():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [4], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x+y) == 4
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    assignment = {'A': 0}

    assert mrv(assignment, csp) == 'B'

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4], 'C': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert (mrv(assignment, csp) == 'B' or
            mrv(assignment, csp) == 'C')

    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5, 6], 'C': [0, 1, 2, 3, 4]}
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assert mrv(assignment, csp) == 'C'


def test_backtracking_search():
    assert backtracking_search(australia)
    assert backtracking_search(australia, select_unassigned_variable=mrv)
    assert backtracking_search(australia, order_domain_values=lcv)
    assert backtracking_search(australia, select_unassigned_variable=mrv,
                               order_domain_values=lcv)
    assert backtracking_search(australia, inference=forward_checking)
    assert backtracking_search(australia, inference=mac)
    assert backtracking_search(usa, select_unassigned_variable=mrv,
                               order_domain_values=lcv, inference=mac)


def test_universal_dict():
    d = UniversalDict(42)
    assert d['life'] == 42


def test_parse_neighbours():
    assert parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}


if __name__ == "__main__":
    pytest.main()
