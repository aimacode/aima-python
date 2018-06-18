from planning import *
from utils import expr
from logic import FolKB, conjuncts


def test_action():
    precond = 'At(c, a) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)'
    effect = 'In(c, p) & ~At(c, a)'
    a = Action('Load(c, p, a)', precond, effect)
    args = [expr("C1"), expr("P1"), expr("SFO")]
    assert a.substitute(expr("Load(c, p, a)"), args) == expr("Load(C1, P1, SFO)")
    test_kb = FolKB(conjuncts(expr('At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK) & Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)')))
    assert a.check_precond(test_kb, args)
    a.act(test_kb, args)
    assert test_kb.ask(expr("In(C1, P2)")) is False
    assert test_kb.ask(expr("In(C1, P1)")) is not False
    assert test_kb.ask(expr("Plane(P2)")) is not False
    assert not a.check_precond(test_kb, args)


def test_air_cargo_1():
    p = air_cargo()
    assert p.goal_test() is False
    solution_1 = [expr("Load(C1 , P1, SFO)"),
                expr("Fly(P1, SFO, JFK)"),
                expr("Unload(C1, P1, JFK)"),
                expr("Load(C2, P2, JFK)"),
                expr("Fly(P2, JFK, SFO)"),
                expr("Unload (C2, P2, SFO)")]  

    for action in solution_1:
        p.act(action)

    assert p.goal_test()


def test_air_cargo_2():
    p = air_cargo()
    assert p.goal_test() is False
    solution_2 = [expr("Load(C2, P2, JFK)"),
                 expr("Fly(P2, JFK, SFO)"),
                 expr("Unload (C2, P2, SFO)"),
                 expr("Load(C1 , P1, SFO)"),
                 expr("Fly(P1, SFO, JFK)"),
                 expr("Unload(C1, P1, JFK)")]

    for action in solution_2:
        p.act(action)

    assert p.goal_test()


def test_spare_tire():
    p = spare_tire()
    assert p.goal_test() is False
    solution = [expr("Remove(Flat, Axle)"),
                expr("Remove(Spare, Trunk)"),
                expr("PutOn(Spare, Axle)")]

    for action in solution:
        p.act(action)

    assert p.goal_test()


def test_spare_tire_2():
    p = spare_tire()
    assert p.goal_test() is False
    solution_2 = [expr('Remove(Spare, Trunk)'),
                  expr('Remove(Flat, Axle)'),
                  expr('PutOn(Spare, Axle)')]

    for action in solution_2:
        p.act(action)

    assert p.goal_test()

    
def test_three_block_tower():
    p = three_block_tower()
    assert p.goal_test() is False
    solution = [expr("MoveToTable(C, A)"),
                expr("Move(B, Table, C)"),
                expr("Move(A, Table, B)")]

    for action in solution:
        p.act(action)

    assert p.goal_test()


def test_have_cake_and_eat_cake_too():
    p = have_cake_and_eat_cake_too()
    assert p.goal_test() is False
    solution = [expr("Eat(Cake)"),
                expr("Bake(Cake)")]

    for action in solution:
        p.act(action)

    assert p.goal_test()


def test_shopping_problem():
    p = shopping_problem()
    assert p.goal_test() is False
    solution = [expr('Go(Home, SM)'), 
                expr('Buy(Banana, SM)'), 
                expr('Buy(Milk, SM)'), 
                expr('Go(SM, HW)'), 
                expr('Buy(Drill, HW)')]

    for action in solution:
        p.act(action)

    assert p.goal_test()


def test_graph_call():
    pddl = spare_tire()
    graph = Graph(pddl)

    levels_size = len(graph.levels)
    graph()

    assert levels_size == len(graph.levels) - 1


def test_graphplan():
    spare_tire_solution = spare_tire_graphplan()
    spare_tire_solution = linearize(spare_tire_solution)
    assert expr('Remove(Flat, Axle)') in spare_tire_solution
    assert expr('Remove(Spare, Trunk)') in spare_tire_solution
    assert expr('PutOn(Spare, Axle)') in spare_tire_solution

    cake_solution = have_cake_and_eat_cake_too_graphplan()
    cake_solution = linearize(cake_solution)
    assert expr('Eat(Cake)') in cake_solution
    assert expr('Bake(Cake)') in cake_solution

    air_cargo_solution = air_cargo_graphplan()
    air_cargo_solution = linearize(air_cargo_solution)
    assert expr('Load(C1, P1, SFO)') in air_cargo_solution
    assert expr('Load(C2, P2, JFK)') in air_cargo_solution
    assert expr('Fly(P1, SFO, JFK)') in air_cargo_solution
    assert expr('Fly(P2, JFK, SFO)') in air_cargo_solution
    assert expr('Unload(C1, P1, JFK)') in air_cargo_solution
    assert expr('Unload(C2, P2, SFO)') in air_cargo_solution

    sussman_anomaly_solution = three_block_tower_graphplan()
    sussman_anomaly_solution = linearize(sussman_anomaly_solution)
    assert expr('MoveToTable(C, A)') in sussman_anomaly_solution
    assert expr('Move(B, Table, C)') in sussman_anomaly_solution
    assert expr('Move(A, Table, B)') in sussman_anomaly_solution

    shopping_problem_solution = shopping_graphplan()
    shopping_problem_solution = linearize(shopping_problem_solution)
    assert expr('Go(Home, HW)') in shopping_problem_solution
    assert expr('Go(Home, SM)') in shopping_problem_solution
    assert expr('Buy(Drill, HW)') in shopping_problem_solution
    assert expr('Buy(Banana, SM)') in shopping_problem_solution
    assert expr('Buy(Milk, SM)') in shopping_problem_solution


def test_total_order_planner():
    st = spare_tire()
    possible_solutions = [[expr('Remove(Spare, Trunk)'), expr('Remove(Flat, Axle)'), expr('PutOn(Spare, Axle)')],
                          [expr('Remove(Flat, Axle)'), expr('Remove(Spare, Trunk)'), expr('PutOn(Spare, Axle)')]]
    assert TotalOrderPlanner(st).execute() in possible_solutions

    ac = air_cargo()
    possible_solutions = [[expr('Load(C1, P1, SFO)'), expr('Load(C2, P2, JFK)'), expr('Fly(P1, SFO, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C1, P1, SFO)'), expr('Load(C2, P2, JFK)'), expr('Fly(P1, SFO, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')],
                          [expr('Load(C1, P1, SFO)'), expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C1, P1, SFO)'), expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')],
                          [expr('Load(C2, P2, JFK)'), expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C2, P2, JFK)'), expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')],
                          [expr('Load(C2, P2, JFK)'), expr('Load(C1, P1, SFO)'), expr('Fly(P2, JFK, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C2, P2, JFK)'), expr('Load(C1, P1, SFO)'), expr('Fly(P2, JFK, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')],
                          [expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')],
                          [expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C1, P1, JFK)'), expr('Unload(C2, P2, SFO)')],
                          [expr('Load(C2, P2, JFK)'), expr('Fly(P2, JFK, SFO)'), expr('Load(C1, P1, SFO)'), expr('Fly(P1, SFO, JFK)'), expr('Unload(C2, P2, SFO)'), expr('Unload(C1, P1, JFK)')]
                          ]
    assert TotalOrderPlanner(ac).execute() in possible_solutions

    ss = socks_and_shoes()
    possible_solutions = [[expr('LeftSock'), expr('RightSock'), expr('LeftShoe'), expr('RightShoe')],
                          [expr('LeftSock'), expr('RightSock'), expr('RightShoe'), expr('LeftShoe')],
                          [expr('RightSock'), expr('LeftSock'), expr('LeftShoe'), expr('RightShoe')],
                          [expr('RightSock'), expr('LeftSock'), expr('RightShoe'), expr('LeftShoe')],
                          [expr('LeftSock'), expr('LeftShoe'), expr('RightSock'), expr('RightShoe')],
                          [expr('RightSock'), expr('RightShoe'), expr('LeftSock'), expr('LeftShoe')]
                          ]
    assert TotalOrderPlanner(ss).execute() in possible_solutions


# def test_double_tennis():
#     p = double_tennis_problem
#     assert p.goal_test() is False

#     solution = [expr("Go(A, RightBaseLine, LeftBaseLine)"),
#                 expr("Hit(A, Ball, RightBaseLine)"),
#                 expr("Go(A, LeftNet, RightBaseLine)")]

#     for action in solution:
#         p.act(action)

#     assert p.goal_test()


def test_job_shop_problem():
    p = job_shop_problem()
    assert p.goal_test() is False

    solution = [p.jobs[1][0],
                p.jobs[0][0],
                p.jobs[0][1],
                p.jobs[0][2],
                p.jobs[1][1],
                p.jobs[1][2]]

    for action in solution:
        p.act(action)

    assert p.goal_test()


def test_refinements():
    
    library = {'HLA': ['Go(Home,SFO)','Taxi(Home, SFO)'],
    'steps': [['Taxi(Home, SFO)'],[]],
    'precond': [['At(Home)'],['At(Home)']],
    'effect': [['At(SFO)'],['At(SFO)'],['~At(Home)'],['~At(Home)']]}

    go_SFO = HLA('Go(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
    taxi_SFO = HLA('Go(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')

    prob = Problem('At(Home)', 'At(SFO)', [go_SFO, taxi_SFO])

    result = [i for i in Problem.refinements(go_SFO, prob, library)]
    assert(len(result) == 1)
    assert(result[0].name == 'Taxi')
    assert(result[0].args == (expr('Home'), expr('SFO')))


def pddl_test_case(domain_file, problem_file, expected_solution):
    domain = DomainParser()
    domain.read(domain_file)

    problem = ProblemParser()
    problem.read(problem_file)

    initial_kb = PlanningKB(problem.goals, problem.initial_state)
    planning_actions = [STRIPSAction(name, preconds, effects) for name, preconds, effects in domain.actions]
    prob = PlanningSearchProblem(initial_kb, planning_actions)
    found_solution = astar_search(prob).solution()

    for action, expected_action in zip(found_solution, expected_solution):
        assert(action == expected_action)


def test_pddl_have_cake_and_eat_it_too():
    """ Negative precondition test for total-order planner. """
    pddl_dir = os.path.join(os.getcwd(), '..', 'pddl_files')
    domain_file = pddl_dir + os.sep + 'cake-domain.pddl'
    problem_file = pddl_dir + os.sep + 'cake-problem.pddl'
    expected_solution = [expr('Eat(Cake)'), expr('Bake(Cake)')]
    pddl_test_case(domain_file, problem_file, expected_solution)


def test_pddl_change_flat_tire():
    """ Positive precondition test for total-order planner. """
    pddl_dir = os.path.join(os.getcwd(), '..', 'pddl_files')
    domain_file = pddl_dir + os.sep + 'spare-tire-domain.pddl'
    problem_file = pddl_dir + os.sep + 'spare-tire-problem.pddl'
    expected_solution = [expr('Remove(Spare, Trunk)'), expr('Remove(Flat, Axle)'), expr('Put_on(Spare, Axle)')]
    pddl_test_case(domain_file, problem_file, expected_solution)


def test_pddl_sussman_anomaly():
    """ Verifying correct action substitution for total-order planner. """
    pddl_dir = os.path.join(os.getcwd(), '..', 'pddl_files')
    domain_file = pddl_dir + os.sep + 'blocks-domain.pddl'
    problem_file = pddl_dir + os.sep + 'sussman-anomaly-problem.pddl'
    expected_solution = [expr('Move_to_table(C, A)'), expr('Move(B, Table, C)'), expr('Move(A, Table, B)')]
    pddl_test_case(domain_file, problem_file, expected_solution)

