import pytest
from search import *  # noqa
from random import choice #noqa


romania = GraphProblem('Arad', 'Bucharest', Fig[3, 2])
vacumm_world = GraphProblemStochastic('State_1', ['State_7', 'State_8'], Fig[4, 9])


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

def test_and_or_graph_search():
    def run_plan(state, problem, plan):
        if problem.goal_test(state):
            return True
        if len(plan) is not 2:
            return False
        next_state = choice(problem.result(state, plan[0]))
        return run_plan(next_state, problem, plan[1][next_state])
    plan = and_or_graph_search(vacumm_world)
    assert run_plan('State_1', vacumm_world, plan)


if __name__ == '__main__':
    pytest.main()
