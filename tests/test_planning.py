from planning import *
from utils import expr
from logic import FolKB


def test_action():
    precond = [[expr("P(x)"), expr("Q(y, z)")], [expr("Q(x)")]]
    effect = [[expr("Q(x)")], [expr("P(x)")]]
    a=Action(expr("A(x,y,z)"), precond, effect)
    args = [expr("A"), expr("B"), expr("C")]
    assert a.substitute(expr("P(x, z, y)"), args) == expr("P(A, C, B)")
    test_kb = FolKB([expr("P(A)"), expr("Q(B, C)"), expr("R(D)")])
    assert a.check_precond(test_kb, args)
    a.act(test_kb, args)
    assert test_kb.ask(expr("P(A)")) is False
    assert test_kb.ask(expr("Q(A)")) is not False
    assert test_kb.ask(expr("Q(B, C)")) is not False
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


def test_graph_call():
    pddl = spare_tire()
    negkb = FolKB([expr('At(Flat, Trunk)')])
    graph = Graph(pddl, negkb)

    levels_size = len(graph.levels)
    graph()

    assert levels_size == len(graph.levels) - 1


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

def test_refinements() :
    init = [expr('At(Home)')]
    def goal_test(kb):
        return kb.ask(expr('At(SFO)'))
        
    library = {"HLA": ["Go(Home,SFO)","Taxi(Home, SFO)"],
    "steps": [["Taxi(Home, SFO)"],[]],
    "precond_pos": [["At(Home)"],["At(Home)"]],
    "precond_neg": [[],[]],
    "effect_pos": [["At(SFO)"],["At(SFO)"]],
    "effect_neg": [["At(Home)"],["At(Home)"],]}
    # Go SFO
    precond_pos = [expr("At(Home)")]
    precond_neg = []
    effect_add = [expr("At(SFO)")]
    effect_rem = [expr("At(Home)")]
    go_SFO = HLA(expr("Go(Home,SFO)"),
                      [precond_pos, precond_neg], [effect_add, effect_rem])
    # Taxi SFO
    precond_pos = [expr("At(Home)")]
    precond_neg = []
    effect_add = [expr("At(SFO)")]
    effect_rem = [expr("At(Home)")]
    taxi_SFO = HLA(expr("Go(Home,SFO)"),
                      [precond_pos, precond_neg], [effect_add, effect_rem])
    prob = Problem(init, [go_SFO, taxi_SFO], goal_test)
    result = [i for i in Problem.refinements(go_SFO, prob, library)]
    assert(len(result) == 1)
    assert(result[0].name == "Taxi")
    assert(result[0].args == (expr("Home"), expr("SFO")))
