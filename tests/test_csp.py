import pytest
from utils import failure_test
from csp import *
import random


random.seed("aima-python")


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
    
    domains = {'A': [ 2, 4], 'B': [ 3, 5]}
    constraints = lambda X, x, Y, y: int(x) > int (y)
    removals=[]
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert AC3(csp, removals=removals)


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


def test_unordered_domain_values():
    map_coloring_test = MapColoringCSP(list('123'), 'A: B C; B: C; C: ')
    assignment = None
    assert unordered_domain_values('A', assignment,  map_coloring_test) == ['1', '2', '3']


def test_lcv():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x+y) == 4
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    assignment = {'A': 0}

    var = 'B'

    assert lcv(var, assignment, csp) == [4, 0, 1, 2, 3, 5]
    assignment = {'A': 1, 'C': 3}

    constraints = lambda X, x, Y, y: (x + y) % 2 == 0 and (x + y) < 5
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    assert lcv(var, assignment, csp) == [1, 3, 0, 2, 4, 5]


def test_forward_checking():
    neighbors = parse_neighbors('A: B; B: C; C: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: (x + y) % 2 == 0 and (x + y) < 8
    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    csp.support_pruning()
    A_curr_domains = csp.curr_domains['A']
    C_curr_domains = csp.curr_domains['C']

    var = 'B'
    value = 3
    assignment = {'A': 1, 'C': '3'}
    assert forward_checking(csp, var, value, assignment, None) == True
    assert csp.curr_domains['A'] == A_curr_domains
    assert csp.curr_domains['C'] == C_curr_domains

    assignment = {'C': 3}

    assert forward_checking(csp, var, value, assignment, None) == True
    assert csp.curr_domains['A'] == [1, 3]

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    csp.support_pruning()

    assignment = {}
    assert forward_checking(csp, var, value, assignment, None) == True
    assert csp.curr_domains['A'] == [1, 3]
    assert csp.curr_domains['C'] == [1, 3]

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4, 7], 'C': [0, 1, 2, 3, 4]}
    csp.support_pruning()

    value = 7
    assignment = {}
    assert forward_checking(csp, var, value, assignment, None) == False
    assert (csp.curr_domains['A'] == [] or csp.curr_domains['C'] == [])


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


def test_min_conflicts():
    assert min_conflicts(australia)
    assert min_conflicts(france)

    tests = [(usa, None)] * 3
    assert failure_test(min_conflicts, tests) >= 1/3

    australia_impossible = MapColoringCSP(list('RG'), 'SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ')
    assert min_conflicts(australia_impossible, 1000) is None
    assert min_conflicts(NQueensCSP(2), 1000) is None
    assert min_conflicts(NQueensCSP(3), 1000) is None


def test_nqueens_csp():
    csp = NQueensCSP(8)

    assignment = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    csp.assign(5, 5, assignment)
    assert len(assignment) == 6
    csp.assign(6, 6, assignment)
    assert len(assignment) == 7
    csp.assign(7, 7, assignment)
    assert len(assignment) == 8
    assert assignment[5] == 5
    assert assignment[6] == 6
    assert assignment[7] == 7
    assert csp.nconflicts(3, 2, assignment) == 0
    assert csp.nconflicts(3, 3, assignment) == 0
    assert csp.nconflicts(1, 5, assignment) == 1
    assert csp.nconflicts(7, 5, assignment) == 2
    csp.unassign(1, assignment)
    csp.unassign(2, assignment)
    csp.unassign(3, assignment)
    assert 1 not in assignment
    assert 2 not in assignment
    assert 3 not in assignment

    assignment = {}
    assignment = {0: 0, 1: 1, 2: 4, 3: 1, 4: 6}
    csp.assign(5, 7, assignment)
    assert len(assignment) == 6
    csp.assign(6, 6, assignment)
    assert len(assignment) == 7
    csp.assign(7, 2, assignment)
    assert len(assignment) == 8
    assert assignment[5] == 7
    assert assignment[6] == 6
    assert assignment[7] == 2
    assignment = {0: 0, 1: 1, 2: 4, 3: 1, 4: 6, 5: 7, 6: 6, 7: 2}
    assert csp.nconflicts(7, 7, assignment) == 4
    assert csp.nconflicts(3, 4, assignment) == 0
    assert csp.nconflicts(2, 6, assignment) == 2
    assert csp.nconflicts(5, 5, assignment) == 3
    csp.unassign(4, assignment)
    csp.unassign(5, assignment)
    csp.unassign(6, assignment)
    assert 4 not in assignment
    assert 5 not in assignment
    assert 6 not in assignment

    for n in range(5, 9):
        csp = NQueensCSP(n)
        solution = min_conflicts(csp)
        assert not solution or sorted(solution.values()) == list(range(n))


def test_universal_dict():
    d = UniversalDict(42)
    assert d['life'] == 42


def test_parse_neighbours():
    assert parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}


def test_topological_sort():
    root = 'NT'
    Sort, Parents = topological_sort(australia,root)

    assert Sort == ['NT','SA','Q','NSW','V','WA']
    assert Parents['NT'] == None
    assert Parents['SA'] == 'NT'
    assert Parents['Q'] == 'SA'
    assert Parents['NSW'] == 'Q'
    assert Parents['V'] == 'NSW'
    assert Parents['WA'] == 'SA'


def test_tree_csp_solver():
    australia_small = MapColoringCSP(list('RB'),
                           'NT: WA Q; NSW: Q V')
    tcs = tree_csp_solver(australia_small)
    assert (tcs['NT'] == 'R' and tcs['WA'] == 'B' and tcs['Q'] == 'B' and tcs['NSW'] == 'R' and tcs['V'] == 'B') or \
           (tcs['NT'] == 'B' and tcs['WA'] == 'R' and tcs['Q'] == 'R' and tcs['NSW'] == 'B' and tcs['V'] == 'R')

if __name__ == "__main__":
    pytest.main()
