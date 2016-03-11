import pytest
from search import *  # noqa


romania = GraphProblem('Arad', 'Bucharest', Fig[3, 2])


def test_breadth_first_tree_search():
    assert breadth_first_tree_search(romania).solution() == ['Sibiu',
                                                             'Fagaras',
                                                             'Bucharest']


def test_breadth_first_search():
    assert breadth_first_search(romania).solution() == ['Sibiu', 'Fagaras',
                                                        'Bucharest']


def test_uniform_cost_search():
    assert uniform_cost_search(romania).solution() == ['Sibiu', 'Rimnicu',
                                                       'Pitesti', 'Bucharest']


def test_depth_first_graph_search():
    solution = depth_first_graph_search(romania).solution()
    assert solution[-1] == 'Bucharest'


def test_iterative_deepening_search():
    assert iterative_deepening_search(romania).solution() == ['Sibiu',
                                                              'Fagaras',
                                                              'Bucharest']

if __name__ == '__main__':
    pytest.main()
