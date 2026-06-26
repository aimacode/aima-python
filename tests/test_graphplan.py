from multiprocessing import Process, Queue

import pytest

from planning import *


def test_blocksworld_manual():
    sbw = simple_blocks_world()
    assert sbw.goal_test() is False
    sbw.act(expr('ToTable(A, B)'))
    sbw.act(expr('FromTable(B, A)'))
    assert sbw.goal_test() is False
    sbw.act(expr('FromTable(C, B)'))
    assert sbw.goal_test() is True


def test_logistics_manual():
    init = 'In(C1, R1) & In(C2, D1) & In(C3, D2) & In(R1, D1) & Holding(R1)'
    goal_state = 'In(C2, D3) & In(C3, D3)'
    p = logistics_problem(init, goal_state)
    assert p.goal_test() is False
    p.act(expr('PutDown(R1, C1, D1)'))
    p.act(expr('PickUp(R1, C2, D1)'))
    p.act(expr('Move(R1, D1, D3)'))
    p.act(expr('PutDown(R1, C2, D3)'))
    p.act(expr('Move(R1, D3, D2)'))
    p.act(expr('PickUp(R1, C3, D2)'))
    p.act(expr('Move(R1, D2, D3)'))
    assert p.goal_test() is False
    p.act(expr('PutDown(R1, C3, D3)'))
    assert p.goal_test() is True


def test_generalized_blocksworld_manual():
    """
    Manual test for the generalized blocks world problem constructor.
    This test case involves stacking four blocks (A, B, C, D) into a single tower.
    """
    initial_state = ('On(A, Table) & On(B, Table) & On(C, Table) & On(D, Table) & '
                     'Clear(A) & Clear(B) & Clear(C) & Clear(D)')
    goal_state = 'On(A, B) & On(B, C) & On(C, D)'
    bw_problem = blocks_world(initial_state, goal_state, ['A', 'B', 'C', 'D'])
    assert bw_problem.goal_test() is False
    bw_problem.act(expr('Move(C, Table, D)'))
    assert bw_problem.goal_test() is False
    bw_problem.act(expr('Move(B, Table, C)'))
    assert bw_problem.goal_test() is False
    bw_problem.act(expr('Move(A, Table, B)'))
    assert bw_problem.goal_test() is True


def verify_solution(p):
    sol = Linearize(p).execute()
    assert p.goal_test() is False
    for act in sol:
        p.act(expr(act))
    assert p.goal_test() is True


def test_air_cargo():
    verify_solution(air_cargo())


def test_spare_tire():
    verify_solution(spare_tire())


def test_three_block_tower():
    verify_solution(three_block_tower())


def test_simple_blocks_world():
    verify_solution(simple_blocks_world())


def test_shopping_problem():
    verify_solution(shopping_problem())


def test_socks_and_shoes():
    verify_solution(socks_and_shoes())


def test_have_cake_and_eat_cake_too():
    verify_solution(have_cake_and_eat_cake_too())


@pytest.mark.parametrize('goal_state', [
    'In(C1, D1)',
    'In(C1, D2)',
    'In(C1, D1) & In(R1, D2)',
    'In(R1, D2) & In(C1, D1)',
    'In(C1, D1) & In(C3, R1)',
    'In(C1, D1) & In(C3, R1) & In(R1, D3)',
    'In(C1, D1) & In(R1, D3) & In(C3, R1)',
    'In(C1, D1) & In(C3, D3)',
    'In(C1, D1) & In(R1, D2) & In(C3, R1)',
    'In(C1, D1) & In(C3, R1) & In(R1, D3)',
    'In(C1, D1) & In(C2, D3)',
    'In(C3, D1)',
    'In(C2, D3)',
    'In(C2, D3) & In(C3, D3)',
    'In(C3, D3) & In(C2, D3)',
    'In(C1, D2) & In(C3, D3)',
    'In(C1, D3) & In(C2, D3) & In(C3, D3)',
    'In(C1, D2) & In(C3, D3) & In(C2, D1)',
    'In(C3, D3)',
    'In(C1, D2) & In(C3, D3) & In(C2, D3) & In(R1, D1)'
])
def test_logistics_plan_valid(goal_state):
    """These should yield a valid (non-crashing) plan, even if empty."""
    init = 'In(C1, R1) & In(C2, D1) & In(C3, D2) & In(R1, D1) & Holding(R1)'
    verify_solution(logistics_problem(init, goal_state))


def test_rush_hour_manual_alt_sequence():
    """
    Provides an alternative manual test for the Rush Hour problem.

    This solution is less efficient but still valid. It interleaves the
    movements of different vehicles and includes an unnecessary move to verify
    that the actions correctly modify the game state without breaking the rules.
    """
    problem = rush_hour()
    assert not problem.goal_test()

    # Make an unnecessary move with the BlueCar to show it works.
    problem.act(expr('MoveUpCar(BlueCar, R4, R5, R6, C2)'))
    assert not problem.goal_test(), 'Moving the BlueCar should not solve the puzzle.'

    # Start clearing the main path by moving the GreenTruck down.
    problem.act(expr('MoveDownTruck(GreenTruck, R1, R2, R3, R4, C4)'))

    # Move the RedCar into the newly available space.
    problem.act(expr('MoveRightCar(RedCar, R3, C1, C2, C3)'))
    assert not problem.goal_test()

    # Continue clearing the path.
    problem.act(expr('MoveDownTruck(GreenTruck, R2, R3, R4, R5, C4)'))
    problem.act(expr('MoveRightCar(RedCar, R3, C2, C3, C4)'))
    assert not problem.goal_test()

    # Final moves to solve the puzzle.
    problem.act(expr('MoveDownTruck(GreenTruck, R3, R4, R5, R6, C4)'))
    problem.act(expr('MoveRightCar(RedCar, R3, C3, C4, C5)'))
    problem.act(expr('MoveRightCar(RedCar, R3, C4, C5, C6)'))
    assert problem.goal_test()


def test_rush_hour_optimized():
    verify_solution(rush_hour_optimized())


def test_planner_leveloff():
    def run_planner_in_queue(problem, queue):
        queue.put(Linearize(problem).execute())

    p = blocks_world(
        'On(A, Table) & On(B, Table) & On(C, Table) & Clear(A) & Clear(B) & Clear(C)',
        'On(A, B) & On(B, C) & On(C, A)',
        ['A', 'B', 'C'])

    result_queue = Queue()
    proc = Process(target=run_planner_in_queue, args=(p, result_queue))
    proc.start()
    proc.join(timeout=3)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        assert False  # ran for 3 seconds and did not exit in level-off
    else:
        result = result_queue.get()
        assert result is None or result == [] or result == [[]]


def test_impossible_cake_exits_via_leveloff():
    """Verify that GraphPlan terminates and returns None for the impossible cake problem."""

    def impossible_cake_problem():
        """
        An impossible planning problem to demonstrate GraphPlan's level-off detection.

        The goal is to both Have(Cake) and Eaten(Cake). However, the only available
        action, Eat(Cake), has the effect of ~Have(Cake). The propositions
        Have(Cake) and Eaten(Cake) become mutually exclusive at the first level,
        and the graph quickly levels off, proving the goal is unreachable.
        """
        return PlanningProblem(initial='Have(Cake) & ~Eaten(Cake)',
                               goals='Have(Cake) & Eaten(Cake)',
                               actions=[Action('Eat(Cake)',
                                               precond='Have(Cake)',
                                               effect='Eaten(Cake) & ~Have(Cake)')])

    assert Linearize(impossible_cake_problem()).execute() is None
