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
    planningproblem = spare_tire()
    graph = Graph(planningproblem)

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


def test_linearize_class():
    st = spare_tire()
    possible_solutions = [[expr('Remove(Spare, Trunk)'), expr('Remove(Flat, Axle)'), expr('PutOn(Spare, Axle)')],
                          [expr('Remove(Flat, Axle)'), expr('Remove(Spare, Trunk)'), expr('PutOn(Spare, Axle)')]]
    assert Linearize(st).execute() in possible_solutions

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
    assert Linearize(ac).execute() in possible_solutions

    ss = socks_and_shoes()
    possible_solutions = [[expr('LeftSock'), expr('RightSock'), expr('LeftShoe'), expr('RightShoe')],
                          [expr('LeftSock'), expr('RightSock'), expr('RightShoe'), expr('LeftShoe')],
                          [expr('RightSock'), expr('LeftSock'), expr('LeftShoe'), expr('RightShoe')],
                          [expr('RightSock'), expr('LeftSock'), expr('RightShoe'), expr('LeftShoe')],
                          [expr('LeftSock'), expr('LeftShoe'), expr('RightSock'), expr('RightShoe')],
                          [expr('RightSock'), expr('RightShoe'), expr('LeftSock'), expr('LeftShoe')]
                          ]
    assert Linearize(ss).execute() in possible_solutions


def test_expand_actions():
    assert len(PartialOrderPlanner(spare_tire()).expand_actions()) == 16
    assert len(PartialOrderPlanner(air_cargo()).expand_actions()) == 360
    assert len(PartialOrderPlanner(have_cake_and_eat_cake_too()).expand_actions()) == 2
    assert len(PartialOrderPlanner(socks_and_shoes()).expand_actions()) == 4
    assert len(PartialOrderPlanner(simple_blocks_world()).expand_actions()) == 12
    assert len(PartialOrderPlanner(three_block_tower()).expand_actions()) == 36


def test_find_open_precondition():
    st = spare_tire()
    pop = PartialOrderPlanner(st)
    assert pop.find_open_precondition()[0] == expr('At(Spare, Axle)')
    assert pop.find_open_precondition()[1] == pop.finish
    assert pop.find_open_precondition()[2][0].name == 'PutOn'

    ss = socks_and_shoes()
    pop = PartialOrderPlanner(ss)
    assert (pop.find_open_precondition()[0] == expr('LeftShoeOn') and pop.find_open_precondition()[2][0].name == 'LeftShoe') or (pop.find_open_precondition()[0] == expr('RightShoeOn') and pop.find_open_precondition()[2][0].name == 'RightShoe')
    assert pop.find_open_precondition()[1] == pop.finish

    cp = have_cake_and_eat_cake_too()
    pop = PartialOrderPlanner(cp)
    assert pop.find_open_precondition()[0] == expr('Eaten(Cake)')
    assert pop.find_open_precondition()[1] == pop.finish
    assert pop.find_open_precondition()[2][0].name == 'Eat'


def test_cyclic():
    st = spare_tire()
    pop = PartialOrderPlanner(st)
    graph = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('e', 'c')]
    assert not pop.cyclic(graph)

    graph =  [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('e', 'c'), ('e', 'b')]
    assert pop.cyclic(graph)

    graph = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('e', 'c'), ('b', 'e'), ('a', 'e')]
    assert not pop.cyclic(graph)

    graph = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('e', 'c'), ('e', 'b'), ('b', 'e'), ('a', 'e')]
    assert pop.cyclic(graph)


def test_partial_order_planner():
    ss = socks_and_shoes()
    pop = PartialOrderPlanner(ss)
    constraints, causal_links = pop.execute(display=False)
    plan = list(reversed(list(pop.toposort(pop.convert(pop.constraints)))))
    assert list(plan[0])[0].name == 'Start'
    assert (list(plan[1])[0].name == 'LeftSock' and list(plan[1])[1].name == 'RightSock') or (list(plan[1])[0].name == 'RightSock' and list(plan[1])[1].name == 'LeftSock')
    assert (list(plan[2])[0].name == 'LeftShoe' and list(plan[2])[1].name == 'RightShoe') or (list(plan[2])[0].name == 'RightShoe' and list(plan[2])[1].name == 'LeftShoe')
    assert list(plan[3])[0].name == 'Finish'


def test_double_tennis():
    p = double_tennis_problem()
    assert not goal_test(p.goals, p.init)

    solution = [expr("Go(A, RightBaseLine, LeftBaseLine)"),
                expr("Hit(A, Ball, RightBaseLine)"),
                expr("Go(A, LeftNet, RightBaseLine)")]

    for action in solution:
        p.act(action)

    assert goal_test(p.goals, p.init)


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


# hierarchies
library_1 = {
        'HLA': ['Go(Home,SFO)', 'Go(Home,SFO)', 'Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)', 'Taxi(Home, SFO)'],
        'steps': [['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'], ['Taxi(Home, SFO)'], [], [], []],
        'precond': [['At(Home) & Have(Car)'], ['At(Home)'], ['At(Home) & Have(Car)'], ['At(SFOLongTermParking)'], ['At(Home)']],
        'effect': [['At(SFO) & ~At(Home)'], ['At(SFO) & ~At(Home) & ~Have(Cash)'], ['At(SFOLongTermParking) & ~At(Home)'], ['At(SFO) & ~At(LongTermParking)'], ['At(SFO) & ~At(Home) & ~Have(Cash)']] }


library_2 = {
        'HLA': ['Go(Home,SFO)', 'Go(Home,SFO)', 'Bus(Home, MetroStop)', 'Metro(MetroStop, SFO)' , 'Metro(MetroStop, SFO)', 'Metro1(MetroStop, SFO)', 'Metro2(MetroStop, SFO)'  ,'Taxi(Home, SFO)'],
        'steps': [['Bus(Home, MetroStop)', 'Metro(MetroStop, SFO)'], ['Taxi(Home, SFO)'], [], ['Metro1(MetroStop, SFO)'], ['Metro2(MetroStop, SFO)'],[],[],[]],
        'precond': [['At(Home)'], ['At(Home)'], ['At(Home)'], ['At(MetroStop)'], ['At(MetroStop)'],['At(MetroStop)'], ['At(MetroStop)'] ,['At(Home) & Have(Cash)']],
        'effect': [['At(SFO) & ~At(Home)'], ['At(SFO) & ~At(Home) & ~Have(Cash)'], ['At(MetroStop) & ~At(Home)'], ['At(SFO) & ~At(MetroStop)'], ['At(SFO) & ~At(MetroStop)'], ['At(SFO) & ~At(MetroStop)'] , ['At(SFO) & ~At(MetroStop)'] ,['At(SFO) & ~At(Home) & ~Have(Cash)']] 
        }


# HLA's
go_SFO = HLA('Go(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
taxi_SFO = HLA('Taxi(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home) & ~Have(Cash)')
drive_SFOLongTermParking = HLA('Drive(Home, SFOLongTermParking)', 'At(Home) & Have(Car)','At(SFOLongTermParking) & ~At(Home)' )
shuttle_SFO = HLA('Shuttle(SFOLongTermParking, SFO)', 'At(SFOLongTermParking)', 'At(SFO) & ~At(LongTermParking)')

# Angelic HLA's
angelic_opt_description = Angelic_HLA('Go(Home, SFO)', precond = 'At(Home)', effect ='$+At(SFO) & $-At(Home)' ) 
angelic_pes_description = Angelic_HLA('Go(Home, SFO)', precond = 'At(Home)', effect ='$+At(SFO) & ~At(Home)' )

# Angelic Nodes
plan1 = Angelic_Node('At(Home)', None, [angelic_opt_description], [angelic_pes_description]) 
plan2 = Angelic_Node('At(Home)', None, [taxi_SFO])
plan3 = Angelic_Node('At(Home)', None, [drive_SFOLongTermParking, shuttle_SFO])

# Problems
prob_1 = Problem('At(Home) & Have(Cash) & Have(Car) ', 'At(SFO) & Have(Cash)', [go_SFO, taxi_SFO, drive_SFOLongTermParking,shuttle_SFO])

initialPlan = [Angelic_Node(prob_1.init, None, [angelic_opt_description], [angelic_pes_description])] 


def test_refinements():
    
    prob = Problem('At(Home) & Have(Car)', 'At(SFO)', [go_SFO])
    result = [i for i in Problem.refinements(go_SFO, prob, library_1)]
            
    assert(result[0][0].name == drive_SFOLongTermParking.name)
    assert(result[0][0].args == drive_SFOLongTermParking.args)
    assert(result[0][0].precond == drive_SFOLongTermParking.precond)
    assert(result[0][0].effect == drive_SFOLongTermParking.effect)

    assert(result[0][1].name == shuttle_SFO.name)
    assert(result[0][1].args == shuttle_SFO.args)
    assert(result[0][1].precond == shuttle_SFO.precond)
    assert(result[0][1].effect == shuttle_SFO.effect)


    assert(result[1][0].name == taxi_SFO.name)
    assert(result[1][0].args == taxi_SFO.args)
    assert(result[1][0].precond == taxi_SFO.precond)
    assert(result[1][0].effect == taxi_SFO.effect)


def test_hierarchical_search(): 

    #test_1
    prob_1 = Problem('At(Home) & Have(Cash) & Have(Car) ', 'At(SFO) & Have(Cash)', [go_SFO])

    solution = Problem.hierarchical_search(prob_1, library_1)

    assert( len(solution) == 2 )

    assert(solution[0].name == drive_SFOLongTermParking.name)
    assert(solution[0].args == drive_SFOLongTermParking.args) 

    assert(solution[1].name == shuttle_SFO.name)
    assert(solution[1].args == shuttle_SFO.args)
    
    #test_2
    solution_2 = Problem.hierarchical_search(prob_1, library_2)

    assert( len(solution_2) == 2 )

    assert(solution_2[0].name == 'Bus')
    assert(solution_2[0].args == (expr('Home'), expr('MetroStop'))) 

    assert(solution_2[1].name == 'Metro1')
    assert(solution_2[1].args == (expr('MetroStop'), expr('SFO')))


def test_convert_angelic_HLA():
    """ 
    Converts angelic HLA's into expressions that correspond to their actions
    ~ : Delete (Not)
    $+ : Possibly add (PosYes)
    $-: Possibly delete (PosNo)
    $$: Possibly add / delete (PosYesNo)
    """
    ang1 = Angelic_HLA('Test', precond = None, effect = '~A')
    ang2 = Angelic_HLA('Test', precond = None, effect = '$+A')
    ang3 = Angelic_HLA('Test', precond = None, effect = '$-A')
    ang4 = Angelic_HLA('Test', precond = None, effect = '$$A')

    assert(ang1.convert(ang1.effect) == [expr('NotA')])
    assert(ang2.convert(ang2.effect) == [expr('PosYesA')])
    assert(ang3.convert(ang3.effect) == [expr('PosNotA')])
    assert(ang4.convert(ang4.effect) == [expr('PosYesNotA')])


def test_is_primitive():
    """
    Tests if a plan is consisted out of primitive HLA's (angelic HLA's)
    """
    assert(not Problem.is_primitive(plan1, library_1))
    assert(Problem.is_primitive(plan2, library_1))
    assert(Problem.is_primitive(plan3, library_1))
    

def test_angelic_action():
    """ 
    Finds the HLA actions that correspond to the HLA actions with angelic semantics 

    h1 : precondition positive: B                                  _______  (add A) or (add A and remove B)
         effect: add A and possibly remove B

    h2 : precondition positive: A                                  _______ (add A and add C) or (delete A and add C) or (add C) or (add A and delete C) or 
         effect: possibly add/remove A and possibly add/remove  C          (delete A and delete C) or (delete C) or (add A) or (delete A) or [] 

    """
    h_1 = Angelic_HLA( expr('h1'), 'B' , 'A & $-B')
    h_2 = Angelic_HLA( expr('h2'), 'A', '$$A & $$C')
    action_1 = Angelic_HLA.angelic_action(h_1)
    action_2 = Angelic_HLA.angelic_action(h_2)
    
    assert ([a.effect for a in action_1] == [ [expr('A'),expr('NotB')], [expr('A')]] )
    assert ([a.effect for a in action_2] == [[expr('A') , expr('C')], [expr('NotA'),  expr('C')], [expr('C')], [expr('A'), expr('NotC')], [expr('NotA'), expr('NotC')], [expr('NotC')], [expr('A')], [expr('NotA')], [None] ] )


def test_optimistic_reachable_set():
    """
    Find optimistic reachable set given a problem initial state and a plan
    """
    h_1 = Angelic_HLA( 'h1', 'B' , '$+A & $-B ')
    h_2 = Angelic_HLA( 'h2', 'A', '$$A & $$C')
    f_1 = HLA('h1', 'B', 'A & ~B')
    f_2 = HLA('h2', 'A', 'A & C')
    problem = Problem('B', 'A', [f_1,f_2] )
    plan = Angelic_Node(problem.init, None, [h_1,h_2], [h_1,h_2])
    opt_reachable_set = Problem.reach_opt(problem.init, plan )
    assert(opt_reachable_set[1] == [[expr('A'), expr('NotB')], [expr('NotB')],[expr('B'), expr('A')], [expr('B')]])
    assert( problem.intersects_goal(opt_reachable_set) )


def test_pesssimistic_reachable_set():
    """
    Find pessimistic reachable set given a problem initial state and a plan
    """
    h_1 = Angelic_HLA( 'h1', 'B' , '$+A & $-B ') 
    h_2 = Angelic_HLA( 'h2', 'A', '$$A & $$C')
    f_1 = HLA('h1', 'B', 'A & ~B')
    f_2 = HLA('h2', 'A', 'A & C')
    problem = Problem('B', 'A', [f_1,f_2] )
    plan = Angelic_Node(problem.init, None, [h_1,h_2], [h_1,h_2])
    pes_reachable_set = Problem.reach_pes(problem.init, plan )
    assert(pes_reachable_set[1] == [[expr('A'), expr('NotB')], [expr('NotB')],[expr('B'), expr('A')], [expr('B')]])
    assert(problem.intersects_goal(pes_reachable_set))


def test_find_reachable_set():
    h_1 = Angelic_HLA( 'h1', 'B' , '$+A & $-B ') 
    f_1 = HLA('h1', 'B', 'A & ~B')
    problem = Problem('B', 'A', [f_1] )
    plan = Angelic_Node(problem.init, None, [h_1], [h_1])
    reachable_set = {0: [problem.init]}
    action_description = [h_1]

    reachable_set = Problem.find_reachable_set(reachable_set, action_description)
    assert(reachable_set[1] == [[expr('A'), expr('NotB')], [expr('NotB')],[expr('B'), expr('A')], [expr('B')]])



def test_intersects_goal():    
    problem_1 = Problem('At(SFO)', 'At(SFO)', [])
    problem_2 =  Problem('At(Home) & Have(Cash) & Have(Car) ', 'At(SFO) & Have(Cash)', [])   
    reachable_set_1 = {0: [problem_1.init]}
    reachable_set_2 = {0: [problem_2.init]}

    assert(Problem.intersects_goal(problem_1, reachable_set_1))
    assert(not Problem.intersects_goal(problem_2, reachable_set_2))


def test_making_progress():
    """
    function not yet implemented
    """
    
    intialPlan_1 = [Angelic_Node(prob_1.init, None, [angelic_opt_description], [angelic_pes_description]), 
            Angelic_Node(prob_1.init, None, [angelic_pes_description], [angelic_pes_description]) ]

    plan_1 = Angelic_Node(prob_1.init, None, [angelic_opt_description], [angelic_pes_description])

    assert(not Problem.making_progress(plan_1, initialPlan))

def test_angelic_search(): 
    """
    Test angelic search for problem, hierarchy, initialPlan
    """
    #test_1
    solution = Problem.angelic_search(prob_1, library_1, initialPlan)

    assert( len(solution) == 2 )

    assert(solution[0].name == drive_SFOLongTermParking.name)
    assert(solution[0].args == drive_SFOLongTermParking.args) 

    assert(solution[1].name == shuttle_SFO.name)
    assert(solution[1].args == shuttle_SFO.args)
    

    #test_2
    solution_2 = Problem.angelic_search(prob_1, library_2, initialPlan)

    assert( len(solution_2) == 2 )

    assert(solution_2[0].name == 'Bus')
    assert(solution_2[0].args == (expr('Home'), expr('MetroStop'))) 

    assert(solution_2[1].name == 'Metro1')
    assert(solution_2[1].args == (expr('MetroStop'), expr('SFO')))
    



