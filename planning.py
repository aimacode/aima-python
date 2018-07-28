"""Planning (Chapters 10-11)
"""

import copy
import itertools
from search import Node
from utils import Expr, expr, first
from logic import FolKB, conjuncts, unify
from collections import deque
from functools import reduce as _reduce


class PlanningProblem:
    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, init, goals, actions):
        self.init = self.convert(init)
        self.goals = self.convert(goals)
        self.actions = actions

    def convert(self, clauses):
        """Converts strings into exprs"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            clauses = clauses

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.init for goal in self.goals)

    def act(self, action):
        """
        Performs the action given as argument.
        Note that action is an Expr like expr('Remove(Glass, Table)') or expr('Eat(Sandwich)')
        """       
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.init, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.init = list_action(self.init, args).clauses


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PlanningProblem.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negative literals.
    Negative preconditions and effects are defined by adding a 'Not' before the name of the clause
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond)
        self.effect = self.convert(effect)

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, Expr(self.name, *self.args))

    def convert(self, clauses):
        """Converts strings into Exprs"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propositional symbol"""

        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""

        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's knowledge base"""

        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            kb.tell(self.substitute(clause, args))
            if clause.op[:3] == 'Not':
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:    
                    kb.retract(self.substitute(new_clause, args))

        return kb


def goal_test(goals, state):
    """Generic goal testing helper function"""

    if isinstance(state, list):
        kb = FolKB(state)
    else:
        kb = state
    return all(kb.ask(q) is not False for q in goals)


def air_cargo():
    """
    [Figure 10.1] AIR-CARGO-PROBLEM

    An air-cargo shipment problem for delivering cargo to different locations,
    given the starting location and airplanes.

    Example:
    >>> from planning import *
    >>> ac = air_cargo()
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Load(C2, P2, JFK)'))
    >>> ac.act(expr('Load(C1, P1, SFO)'))
    >>> ac.act(expr('Fly(P1, SFO, JFK)'))
    >>> ac.act(expr('Fly(P2, JFK, SFO)'))
    >>> ac.act(expr('Unload(C2, P2, SFO)'))
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Unload(C1, P1, JFK)'))
    >>> ac.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK) & Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)', 
                goals='At(C1, JFK) & At(C2, SFO)',
                actions=[Action('Load(c, p, a)', 
                                precond='At(c, a) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)',
                                effect='In(c, p) & ~At(c, a)'),
                         Action('Unload(c, p, a)',
                                precond='In(c, p) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)',
                                effect='At(c, a) & ~In(c, p)'),
                         Action('Fly(p, f, to)',
                                precond='At(p, f) & Plane(p) & Airport(f) & Airport(to)',
                                effect='At(p, to) & ~At(p, f)')])


def spare_tire():
    """[Figure 10.2] SPARE-TIRE-PROBLEM

    A problem involving changing the flat tire of a car
    with a spare tire from the trunk.

    Example:
    >>> from planning import *
    >>> st = spare_tire()
    >>> st.goal_test()
    False
    >>> st.act(expr('Remove(Spare, Trunk)'))
    >>> st.act(expr('Remove(Flat, Axle)'))
    >>> st.goal_test()
    False
    >>> st.act(expr('PutOn(Spare, Axle)'))
    >>> st.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='Tire(Flat) & Tire(Spare) & At(Flat, Axle) & At(Spare, Trunk)',
                goals='At(Spare, Axle) & At(Flat, Ground)',
                actions=[Action('Remove(obj, loc)',
                                precond='At(obj, loc)',
                                effect='At(obj, Ground) & ~At(obj, loc)'),
                         Action('PutOn(t, Axle)',
                                precond='Tire(t) & At(t, Ground) & ~At(Flat, Axle)',
                                effect='At(t, Axle) & ~At(t, Ground)'),
                         Action('LeaveOvernight',
                                precond='',
                                effect='~At(Spare, Ground) & ~At(Spare, Axle) & ~At(Spare, Trunk) & \
                                        ~At(Flat, Ground) & ~At(Flat, Axle) & ~At(Flat, Trunk)')])


def three_block_tower():
    """
    [Figure 10.3] THREE-BLOCK-TOWER

    A blocks-world problem of stacking three blocks in a certain configuration,
    also known as the Sussman Anomaly.

    Example:
    >>> from planning import *
    >>> tbt = three_block_tower()
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('MoveToTable(C, A)'))
    >>> tbt.act(expr('Move(B, Table, C)'))
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('Move(A, Table, B)'))
    >>> tbt.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='On(A, Table) & On(B, Table) & On(C, A) & Block(A) & Block(B) & Block(C) & Clear(B) & Clear(C)',
                goals='On(A, B) & On(B, C)',
                actions=[Action('Move(b, x, y)',
                                precond='On(b, x) & Clear(b) & Clear(y) & Block(b) & Block(y)',
                                effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)'),
                         Action('MoveToTable(b, x)',
                                precond='On(b, x) & Clear(b) & Block(b)',
                                effect='On(b, Table) & Clear(x) & ~On(b, x)')])


def simple_blocks_world():
    """
    SIMPLE-BLOCKS-WORLD

    A simplified definition of the Sussman Anomaly problem.

    Example:
    >>> from planning import *
    >>> sbw = simple_blocks_world()
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('ToTable(A, B)'))
    >>> sbw.act(expr('FromTable(B, A)'))
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('FromTable(C, B)'))
    >>> sbw.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='On(A, B) & Clear(A) & OnTable(B) & OnTable(C) & Clear(C)',
                goals='On(B, A) & On(C, B)',
                actions=[Action('ToTable(x, y)',
                                precond='On(x, y) & Clear(x)',
                                effect='~On(x, y) & Clear(y) & OnTable(x)'),
                         Action('FromTable(y, x)',
                                precond='OnTable(y) & Clear(y) & Clear(x)',
                                effect='~OnTable(y) & ~Clear(x) & On(y, x)')])


def have_cake_and_eat_cake_too():
    """
    [Figure 10.7] CAKE-PROBLEM

    A problem where we begin with a cake and want to 
    reach the state of having a cake and having eaten a cake.
    The possible actions include baking a cake and eating a cake.

    Example:
    >>> from planning import *
    >>> cp = have_cake_and_eat_cake_too()
    >>> cp.goal_test()
    False
    >>> cp.act(expr('Eat(Cake)'))
    >>> cp.goal_test()
    False
    >>> cp.act(expr('Bake(Cake)'))
    >>> cp.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='Have(Cake)',
                goals='Have(Cake) & Eaten(Cake)',
                actions=[Action('Eat(Cake)',
                                precond='Have(Cake)',
                                effect='Eaten(Cake) & ~Have(Cake)'),
                         Action('Bake(Cake)',
                                precond='~Have(Cake)',
                                effect='Have(Cake)')])


def shopping_problem():
    """
    SHOPPING-PROBLEM

    A problem of acquiring some items given their availability at certain stores.

    Example:
    >>> from planning import *
    >>> sp = shopping_problem()
    >>> sp.goal_test()
    False
    >>> sp.act(expr('Go(Home, HW)'))
    >>> sp.act(expr('Buy(Drill, HW)'))
    >>> sp.act(expr('Go(HW, SM)'))
    >>> sp.act(expr('Buy(Banana, SM)'))
    >>> sp.goal_test()
    False
    >>> sp.act(expr('Buy(Milk, SM)'))
    >>> sp.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='At(Home) & Sells(SM, Milk) & Sells(SM, Banana) & Sells(HW, Drill)',
                goals='Have(Milk) & Have(Banana) & Have(Drill)', 
                actions=[Action('Buy(x, store)',
                                precond='At(store) & Sells(store, x)',
                                effect='Have(x)'),
                         Action('Go(x, y)',
                                precond='At(x)',
                                effect='At(y) & ~At(x)')])


def socks_and_shoes():
    """
    SOCKS-AND-SHOES-PROBLEM

    A task of wearing socks and shoes on both feet

    Example:
    >>> from planning import *
    >>> ss = socks_and_shoes()
    >>> ss.goal_test()
    False
    >>> ss.act(expr('RightSock'))
    >>> ss.act(expr('RightShoe'))
    >>> ss.act(expr('LeftSock'))
    >>> ss.goal_test()
    False
    >>> ss.act(expr('LeftShoe'))
    >>> ss.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='',
                goals='RightShoeOn & LeftShoeOn',
                actions=[Action('RightShoe',
                                precond='RightSockOn',
                                effect='RightShoeOn'),
                        Action('RightSock',
                                precond='',
                                effect='RightSockOn'),
                        Action('LeftShoe',
                                precond='LeftSockOn',
                                effect='LeftShoeOn'),
                        Action('LeftSock',
                                precond='',
                                effect='LeftSockOn')])


def double_tennis_problem():
    """
    [Figure 11.10] DOUBLE-TENNIS-PROBLEM

    A multiagent planning problem involving two partner tennis players
    trying to return an approaching ball and repositioning around in the court.

    Example:
    >>> from planning import *
    >>> dtp = double_tennis_problem()
    >>> goal_test(dtp.goals, dtp.init)
    False
    >>> dtp.act(expr('Go(A, RightBaseLine, LeftBaseLine)'))
    >>> dtp.act(expr('Hit(A, Ball, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.init)
    False
    >>> dtp.act(expr('Go(A, LeftNet, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.init)
    True
    >>>
    """

    return PlanningProblem(init='At(A, LeftBaseLine) & At(B, RightNet) & Approaching(Ball, RightBaseLine) & Partner(A, B) & Partner(B, A)',
                             goals='Returned(Ball) & At(a, LeftNet) & At(a, RightNet)',
                             actions=[Action('Hit(actor, Ball, loc)',
                                             precond='Approaching(Ball, loc) & At(actor, loc)',
                                             effect='Returned(Ball)'),
                                      Action('Go(actor, to, loc)', 
                                             precond='At(actor, loc)',
                                             effect='At(actor, to) & ~At(actor, loc)')])


class Level:
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, kb):
        """Initializes variables to hold state and action details of a level"""

        self.kb = kb
        # current state
        self.current_state = kb.clauses
        # current action to state link
        self.current_action_links = {}
        # current state to action link
        self.current_state_links = {}
        # current action to next state link
        self.next_action_links = {}
        # next state to current action link
        self.next_state_links = {}
        # mutually exclusive actions
        self.mutex = []

    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()

    def separate(self, e):
        """Separates an iterable of elements into positive and negative parts"""

        positive = []
        negative = []
        for clause in e:
            if clause.op[:3] == 'Not':
                negative.append(clause)
            else:
                positive.append(clause)
        return positive, negative

    def find_mutex(self):
        """Finds mutually exclusive actions"""

        # Inconsistent effects
        pos_nsl, neg_nsl = self.separate(self.next_state_links)

        for negeff in neg_nsl:
            new_negeff = Expr(negeff.op[3:], *negeff.args)
            for poseff in pos_nsl:
                if new_negeff == poseff:
                    for a in self.next_state_links[poseff]:
                        for b in self.next_state_links[negeff]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Interference will be calculated with the last step
        pos_csl, neg_csl = self.separate(self.current_state_links)

        # Competing needs
        for posprecond in pos_csl:
            for negprecond in neg_csl:
                new_negprecond = Expr(negprecond.op[3:], *negprecond.args)
                if new_negprecond == posprecond:
                    for a in self.current_state_links[posprecond]:
                        for b in self.current_state_links[negprecond]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Inconsistent support
        state_mutex = []
        for pair in self.mutex:
            next_state_0 = self.next_action_links[list(pair)[0]]
            if len(pair) == 2:
                next_state_1 = self.next_action_links[list(pair)[1]]
            else:
                next_state_1 = self.next_action_links[list(pair)[0]]
            if (len(next_state_0) == 1) and (len(next_state_1) == 1):
                state_mutex.append({next_state_0[0], next_state_1[0]})
        
        self.mutex = self.mutex + state_mutex

    def build(self, actions, objects):
        """Populates the lists and dictionaries containing the state action dependencies"""

        for clause in self.current_state:
            p_expr = Expr('P' + clause.op, *clause.args)
            self.current_action_links[p_expr] = [clause]
            self.next_action_links[p_expr] = [clause]
            self.current_state_links[clause] = [p_expr]
            self.next_state_links[clause] = [p_expr]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.kb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower():
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links[new_action] = []

                    for clause in a.precond:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links[new_action].append(new_clause)
                        if new_clause in self.current_state_links:
                            self.current_state_links[new_clause].append(new_action)
                        else:
                            self.current_state_links[new_clause] = [new_action]
                   
                    self.next_action_links[new_action] = []
                    for clause in a.effect:
                        new_clause = a.substitute(clause, arg)

                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links:
                            self.next_state_links[new_clause].append(new_action)
                        else:
                            self.next_state_links[new_clause] = [new_action]

    def perform_actions(self):
        """Performs the necessary actions and returns a new Level"""

        new_kb = FolKB(list(set(self.next_state_links.keys())))
        return Level(new_kb)


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, planningproblem):
        self.planningproblem = planningproblem
        self.kb = FolKB(planningproblem.init)
        self.levels = [Level(self.kb)]
        self.objects = set(arg for clause in self.kb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def expand_graph(self):
        """Expands the graph by a level"""

        last_level = self.levels[-1]
        last_level(self.planningproblem.actions, self.objects)
        self.levels.append(last_level.perform_actions())

    def non_mutex_goals(self, goals, index):
        """Checks whether the goals are mutually exclusive"""

        goal_perm = itertools.combinations(goals, 2)
        for g in goal_perm:
            if set(g) in self.levels[index].mutex:
                return False
        return True


class GraphPlan:
    """
    Class for formulation GraphPlan algorithm
    Constructs a graph of state and action space
    Returns solution for the planning problem
    """

    def __init__(self, planningproblem):
        self.graph = Graph(planningproblem)
        self.nogoods = []
        self.solution = []

    def check_leveloff(self):
        """Checks if the graph has levelled off"""

        check = (set(self.graph.levels[-1].current_state) == set(self.graph.levels[-2].current_state))

        if check:
            return True

    def extract_solution(self, goals, index):
        """Extracts the solution"""

        level = self.graph.levels[index]    
        if not self.graph.non_mutex_goals(goals, index):
            self.nogoods.append((level, goals))
            return

        level = self.graph.levels[index - 1]    

        # Create all combinations of actions that satisfy the goal    
        actions = []
        for goal in goals:
            actions.append(level.next_state_links[goal])    

        all_actions = list(itertools.product(*actions))    

        # Filter out non-mutex actions
        non_mutex_actions = []    
        for action_tuple in all_actions:
            action_pairs = itertools.combinations(list(set(action_tuple)), 2)        
            non_mutex_actions.append(list(set(action_tuple)))        
            for pair in action_pairs:            
                if set(pair) in level.mutex:
                    non_mutex_actions.pop(-1)
                    break
    

        # Recursion
        for action_list in non_mutex_actions:        
            if [action_list, index] not in self.solution:
                self.solution.append([action_list, index])

                new_goals = []
                for act in set(action_list):                
                    if act in level.current_action_links:
                        new_goals = new_goals + level.current_action_links[act]

                if abs(index) + 1 == len(self.graph.levels):
                    return
                elif (level, new_goals) in self.nogoods:
                    return
                else:
                    self.extract_solution(new_goals, index - 1)

        # Level-Order multiple solutions
        solution = []
        for item in self.solution:
            if item[1] == -1:
                solution.append([])
                solution[-1].append(item[0])
            else:
                solution[-1].append(item[0])

        for num, item in enumerate(solution):
            item.reverse()
            solution[num] = item

        return solution

    def goal_test(self, kb):
        return all(kb.ask(q) is not False for q in self.graph.planningproblem.goals)

    def execute(self):
        """Executes the GraphPlan algorithm for the given problem"""

        while True:
            self.graph.expand_graph()
            if (self.goal_test(self.graph.levels[-1].kb) and self.graph.non_mutex_goals(self.graph.planningproblem.goals, -1)):
                solution = self.extract_solution(self.graph.planningproblem.goals, -1)
                if solution:
                    return solution
            
            if len(self.graph.levels) >= 2 and self.check_leveloff():
                return None


class Linearize:

    def __init__(self, planningproblem):
        self.planningproblem = planningproblem

    def filter(self, solution):
        """Filter out persistence actions from a solution"""

        new_solution = []
        for section in solution[0]:
            new_section = []
            for operation in section:
                if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                    new_section.append(operation)
            new_solution.append(new_section)
        return new_solution

    def orderlevel(self, level, planningproblem):
        """Return valid linear order of actions for a given level"""

        for permutation in itertools.permutations(level):
            temp = copy.deepcopy(planningproblem)
            count = 0
            for action in permutation:
                try:
                    temp.act(action)
                    count += 1
                except:
                    count = 0
                    temp = copy.deepcopy(planningproblem)
                    break
            if count == len(permutation):
                return list(permutation), temp
        return None

    def execute(self):
        """Finds total-order solution for a planning graph"""

        graphplan_solution = GraphPlan(self.planningproblem).execute()
        filtered_solution = self.filter(graphplan_solution)
        ordered_solution = []
        planningproblem = self.planningproblem
        for level in filtered_solution:
            level_solution, planningproblem = self.orderlevel(level, planningproblem)
            for element in level_solution:
                ordered_solution.append(element)

        return ordered_solution


def linearize(solution):
    """Converts a level-ordered solution into a linear solution"""

    linear_solution = []
    for section in solution[0]:
        for operation in section:
            if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                linear_solution.append(operation)

    return linear_solution


'''
[Section 10.13] PARTIAL-ORDER-PLANNER

Partially ordered plans are created by a search through the space of plans
rather than a search through the state space. It views planning as a refinement of partially ordered plans.
A partially ordered plan is defined by a set of actions and a set of constraints of the form A < B,
which denotes that action A has to be performed before action B.
To summarize the working of a partial order planner,
1. An open precondition is selected (a sub-goal that we want to achieve).
2. An action that fulfils the open precondition is chosen.
3. Temporal constraints are updated.
4. Existing causal links are protected. Protection is a method that checks if the causal links conflict
   and if they do, temporal constraints are added to fix the threats.
5. The set of open preconditions is updated.
6. Temporal constraints of the selected action and the next action are established.
7. A new causal link is added between the selected action and the owner of the open precondition.
8. The set of new causal links is checked for threats and if found, the threat is removed by either promotion or demotion.
   If promotion or demotion is unable to solve the problem, the planning problem cannot be solved with the current sequence of actions
   or it may not be solvable at all.
9. These steps are repeated until the set of open preconditions is empty.
'''

class PartialOrderPlanner:

    def __init__(self, planningproblem):
        self.planningproblem = planningproblem
        self.initialize()

    def initialize(self):
        """Initialize all variables"""
        self.causal_links = []
        self.start = Action('Start', [], self.planningproblem.init)
        self.finish = Action('Finish', self.planningproblem.goals, [])
        self.actions = set()
        self.actions.add(self.start)
        self.actions.add(self.finish)
        self.constraints = set()
        self.constraints.add((self.start, self.finish))
        self.agenda = set()
        for precond in self.finish.precond:
            self.agenda.add((precond, self.finish))
        self.expanded_actions = self.expand_actions()

    def expand_actions(self, name=None):
        """Generate all possible actions with variable bindings for precondition selection heuristic"""

        objects = set(arg for clause in self.planningproblem.init for arg in clause.args)
        expansions = []
        action_list = []
        if name is not None:
            for action in self.planningproblem.actions:
                if str(action.name) == name:
                    action_list.append(action)
        else:
            action_list = self.planningproblem.actions

        for action in action_list:
            for permutation in itertools.permutations(objects, len(action.args)):
                bindings = unify(Expr(action.name, *action.args), Expr(action.name, *permutation))
                if bindings is not None:
                    new_args = []
                    for arg in action.args:
                        if arg in bindings:
                            new_args.append(bindings[arg])
                        else:
                            new_args.append(arg)
                    new_expr = Expr(str(action.name), *new_args)
                    new_preconds = []
                    for precond in action.precond:
                        new_precond_args = []
                        for arg in precond.args:
                            if arg in bindings:
                                new_precond_args.append(bindings[arg])
                            else:
                                new_precond_args.append(arg)
                        new_precond = Expr(str(precond.op), *new_precond_args)
                        new_preconds.append(new_precond)
                    new_effects = []
                    for effect in action.effect:
                        new_effect_args = []
                        for arg in effect.args:
                            if arg in bindings:
                                new_effect_args.append(bindings[arg])
                            else:
                                new_effect_args.append(arg)
                        new_effect = Expr(str(effect.op), *new_effect_args)
                        new_effects.append(new_effect)
                    expansions.append(Action(new_expr, new_preconds, new_effects))

        return expansions

    def find_open_precondition(self):
        """Find open precondition with the least number of possible actions"""

        number_of_ways = dict()
        actions_for_precondition = dict()
        for element in self.agenda:
            open_precondition = element[0]
            possible_actions = list(self.actions) + self.expanded_actions
            for action in possible_actions:
                for effect in action.effect:
                    if effect == open_precondition:
                        if open_precondition in number_of_ways:
                            number_of_ways[open_precondition] += 1
                            actions_for_precondition[open_precondition].append(action)
                        else:
                            number_of_ways[open_precondition] = 1
                            actions_for_precondition[open_precondition] = [action]

        number = sorted(number_of_ways, key=number_of_ways.__getitem__)
        
        for k, v in number_of_ways.items():
            if v == 0:
                return None, None, None

        act1 = None
        for element in self.agenda:
            if element[0] == number[0]:
                act1 = element[1]
                break

        if number[0] in self.expanded_actions:
            self.expanded_actions.remove(number[0])

        return number[0], act1, actions_for_precondition[number[0]]

    def find_action_for_precondition(self, oprec):
        """Find action for a given precondition"""

        # either
        #   choose act0 E Actions such that act0 achieves G
        for action in self.actions:
            for effect in action.effect:
                if effect == oprec:
                    return action, 0

        # or
        #   choose act0 E Actions such that act0 achieves G
        for action in self.planningproblem.actions:
            for effect in action.effect:
                if effect.op == oprec.op:
                    bindings = unify(effect, oprec)
                    if bindings is None:
                        break
                    return action, bindings

    def generate_expr(self, clause, bindings):
        """Generate atomic expression from generic expression given variable bindings"""

        new_args = []
        for arg in clause.args:
            if arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)

        try:
            return Expr(str(clause.name), *new_args)
        except:
            return Expr(str(clause.op), *new_args)
        
    def generate_action_object(self, action, bindings):
        """Generate action object given a generic action andvariable bindings"""

        # if bindings is 0, it means the action already exists in self.actions
        if bindings == 0:
            return action

        # bindings cannot be None
        else:
            new_expr = self.generate_expr(action, bindings)
            new_preconds = []
            for precond in action.precond:
                new_precond = self.generate_expr(precond, bindings)
                new_preconds.append(new_precond)
            new_effects = []
            for effect in action.effect:
                new_effect = self.generate_expr(effect, bindings)
                new_effects.append(new_effect)
            return Action(new_expr, new_preconds, new_effects)

    def cyclic(self, graph):
        """Check cyclicity of a directed graph"""

        new_graph = dict()
        for element in graph:
            if element[0] in new_graph:
                new_graph[element[0]].append(element[1])
            else:
                new_graph[element[0]] = [element[1]]

        path = set()

        def visit(vertex):
            path.add(vertex)
            for neighbor in new_graph.get(vertex, ()):
                if neighbor in path or visit(neighbor):
                    return True
            path.remove(vertex)
            return False

        value = any(visit(v) for v in new_graph)
        return value

    def add_const(self, constraint, constraints):
        """Add the constraint to constraints if the resulting graph is acyclic"""

        if constraint[0] == self.finish or constraint[1] == self.start:
            return constraints

        new_constraints = set(constraints)
        new_constraints.add(constraint)

        if self.cyclic(new_constraints):
            return constraints
        return new_constraints

    def is_a_threat(self, precondition, effect):
        """Check if effect is a threat to precondition"""

        if (str(effect.op) == 'Not' + str(precondition.op)) or ('Not' + str(effect.op) == str(precondition.op)):
            if effect.args == precondition.args:
                return True
        return False

    def protect(self, causal_link, action, constraints):
        """Check and resolve threats by promotion or demotion"""

        threat = False
        for effect in action.effect:
            if self.is_a_threat(causal_link[1], effect):
                threat = True
                break

        if action != causal_link[0] and action != causal_link[2] and threat:
            # try promotion
            new_constraints = set(constraints)
            new_constraints.add((action, causal_link[0]))
            if not self.cyclic(new_constraints):
                constraints = self.add_const((action, causal_link[0]), constraints)
            else:
                # try demotion
                new_constraints = set(constraints)
                new_constraints.add((causal_link[2], action))
                if not self.cyclic(new_constraints):
                    constraints = self.add_const((causal_link[2], action), constraints)
                else:
                    # both promotion and demotion fail
                    print('Unable to resolve a threat caused by', action, 'onto', causal_link)
                    return
        return constraints

    def convert(self, constraints):
        """Convert constraints into a dict of Action to set orderings"""

        graph = dict()
        for constraint in constraints:
            if constraint[0] in graph:
                graph[constraint[0]].add(constraint[1])
            else:
                graph[constraint[0]] = set()
                graph[constraint[0]].add(constraint[1])
        return graph

    def toposort(self, graph):
        """Generate topological ordering of constraints"""

        if len(graph) == 0:
            return

        graph = graph.copy()

        for k, v in graph.items():
            v.discard(k)

        extra_elements_in_dependencies = _reduce(set.union, graph.values()) - set(graph.keys())

        graph.update({element:set() for element in extra_elements_in_dependencies})
        while True:
            ordered = set(element for element, dependency in graph.items() if len(dependency) == 0)
            if not ordered:
                break
            yield ordered
            graph = {element: (dependency - ordered) for element, dependency in graph.items() if element not in ordered}
        if len(graph) != 0:
            raise ValueError('The graph is not acyclic and cannot be linearly ordered')

    def display_plan(self):
        """Display causal links, constraints and the plan"""

        print('Causal Links')
        for causal_link in self.causal_links:
            print(causal_link)

        print('\nConstraints')
        for constraint in self.constraints:
            print(constraint[0], '<', constraint[1])

        print('\nPartial Order Plan')
        print(list(reversed(list(self.toposort(self.convert(self.constraints))))))

    def execute(self, display=True):
        """Execute the algorithm"""

        step = 1
        self.tries = 1
        while len(self.agenda) > 0:
            step += 1
            # select <G, act1> from Agenda
            try:
                G, act1, possible_actions = self.find_open_precondition()
            except IndexError:
                print('Probably Wrong')
                break

            act0 = possible_actions[0]
            # remove <G, act1> from Agenda
            self.agenda.remove((G, act1))

            # For actions with variable number of arguments, use least commitment principle
            # act0_temp, bindings = self.find_action_for_precondition(G)
            # act0 = self.generate_action_object(act0_temp, bindings)

            # Actions = Actions U {act0}
            self.actions.add(act0)

            # Constraints = add_const(start < act0, Constraints)
            self.constraints = self.add_const((self.start, act0), self.constraints)

            # for each CL E CausalLinks do
            #   Constraints = protect(CL, act0, Constraints)
            for causal_link in self.causal_links:
                self.constraints = self.protect(causal_link, act0, self.constraints)

            # Agenda = Agenda U {<P, act0>: P is a precondition of act0}
            for precondition in act0.precond:
                self.agenda.add((precondition, act0))

            # Constraints = add_const(act0 < act1, Constraints)
            self.constraints = self.add_const((act0, act1), self.constraints)

            # CausalLinks U {<act0, G, act1>}
            if (act0, G, act1) not in self.causal_links:
                self.causal_links.append((act0, G, act1))

            # for each A E Actions do
            #   Constraints = protect(<act0, G, act1>, A, Constraints)
            for action in self.actions:
                self.constraints = self.protect((act0, G, act1), action, self.constraints)

            if step > 200:
                print('Couldn\'t find a solution')
                return None, None

        if display:
            self.display_plan()
        else:
            return self.constraints, self.causal_links                


def spare_tire_graphplan():
    """Solves the spare tire problem using GraphPlan"""
    return GraphPlan(spare_tire()).execute()

def three_block_tower_graphplan():
    """Solves the Sussman Anomaly problem using GraphPlan"""
    return GraphPlan(three_block_tower()).execute()

def air_cargo_graphplan():
    """Solves the air cargo problem using GraphPlan"""
    return GraphPlan(air_cargo()).execute()

def have_cake_and_eat_cake_too_graphplan():
    """Solves the cake problem using GraphPlan"""
    return [GraphPlan(have_cake_and_eat_cake_too()).execute()[1]]

def shopping_graphplan():
    """Solves the shopping problem using GraphPlan"""
    return GraphPlan(shopping_problem()).execute()

def socks_and_shoes_graphplan():
    """Solves the socks and shoes problem using GraphpPlan"""
    return GraphPlan(socks_and_shoes()).execute()

def simple_blocks_world_graphplan():
    """Solves the simple blocks world problem"""
    return GraphPlan(simple_blocks_world()).execute()


class HLA(Action):
    """
    Define Actions for the real-world (that may be refined further), and satisfy resource
    constraints.
    """
    unique_group = 1

    def __init__(self, action, precond=None, effect=None, duration=0,
                 consume=None, use=None):
        """
        As opposed to actions, to define HLA, we have added constraints.
        duration holds the amount of time required to execute the task
        consumes holds a dictionary representing the resources the task consumes
        uses holds a dictionary representing the resources the task uses
        """
        precond = precond or [None]
        effect = effect or [None]
        super().__init__(action, precond, effect)
        self.duration = duration
        self.consumes = consume or {}
        self.uses = use or {}
        self.completed = False
        # self.priority = -1 #  must be assigned in relation to other HLAs
        # self.job_group = -1 #  must be assigned in relation to other HLAs

    def do_action(self, job_order, available_resources, kb, args):
        """
        An HLA based version of act - along with knowledge base updation, it handles
        resource checks, and ensures the actions are executed in the correct order.
        """
        # print(self.name)
        if not self.has_usable_resource(available_resources):
            raise Exception('Not enough usable resources to execute {}'.format(self.name))
        if not self.has_consumable_resource(available_resources):
            raise Exception('Not enough consumable resources to execute {}'.format(self.name))
        if not self.inorder(job_order):
            raise Exception("Can't execute {} - execute prerequisite actions first".
                            format(self.name))
        kb = super().act(kb, args)  # update knowledge base
        for resource in self.consumes:  # remove consumed resources
            available_resources[resource] -= self.consumes[resource]
        self.completed = True  # set the task status to complete
        return kb

    def has_consumable_resource(self, available_resources):
        """
        Ensure there are enough consumable resources for this action to execute.
        """
        for resource in self.consumes:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.consumes[resource]:
                return False
        return True

    def has_usable_resource(self, available_resources):
        """
        Ensure there are enough usable resources for this action to execute.
        """
        for resource in self.uses:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.uses[resource]:
                return False
        return True

    def inorder(self, job_order):
        """
        Ensure that all the jobs that had to be executed before the current one have been
        successfully executed.
        """
        for jobs in job_order:
            if self in jobs:
                for job in jobs:
                    if job is self:
                        return True
                    if not job.completed:
                        return False
        return True


class Problem(PlanningProblem):
    """
    Define real-world problems by aggregating resources as numerical quantities instead of
    named entities.

    This class is identical to PDLL, except that it overloads the act function to handle
    resource and ordering conditions imposed by HLA as opposed to Action.
    """
    def __init__(self, init, goals, actions, jobs=None, resources=None):
        super().__init__(init, goals, actions)
        self.jobs = jobs
        self.resources = resources or {}

    def act(self, action):
        """
        Performs the HLA given as argument.

        Note that this is different from the superclass action - where the parameter was an
        Expression. For real world problems, an Expr object isn't enough to capture all the
        detail required for executing the action - resources, preconditions, etc need to be
        checked for too.
        """
        args = action.args
        list_action = first(a for a in self.actions if a.name == action.name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action.name))
        self.init = list_action.do_action(self.jobs, self.resources, self.init, args).clauses

    def refinements(hla, state, library):  # refinements may be (multiple) HLA themselves ...
        """
        state is a Problem, containing the current state kb
        library is a dictionary containing details for every possible refinement. eg:
        {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
            ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
            ],
        # empty refinements indicate a primitive action
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
            ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
            ]
        }
        """
        e = Expr(hla.name, hla.args)
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
        for i in indices:
            actions = []
            for j in range(len(library['steps'][i])):
                # find the index of the step [j]  of the HLA 
                index_step = [k for k,x in enumerate(library['HLA']) if x == library['steps'][i][j]][0]
                precond = library['precond'][index_step][0] # preconditions of step [j]
                effect = library['effect'][index_step][0] # effect of step [j]
                actions.append(HLA(library['steps'][i][j], precond, effect))
            yield actions

    def hierarchical_search(problem, hierarchy):
        """
        [Figure 11.5] 'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world problem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details)
        """
        act = Node(problem.init, None, [problem.actions[0]])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            (hla, index) = Problem.find_hla(plan, hierarchy) # finds the first non primitive hla in plan actions
            prefix = plan.action[:index]
            outcome = Problem(Problem.result(problem.init, prefix), problem.goals , problem.actions )
            suffix = plan.action[index+1:]
            if not hla: # hla is None and plan is primitive
                if outcome.goal_test():
                    return plan.action
            else:
                for sequence in Problem.refinements(hla, outcome, hierarchy): # find refinements
                    frontier.append(Node(outcome.init, plan, prefix + sequence+ suffix))

    def result(state, actions):
        """The outcome of applying an action to the current problem"""
        for a in actions: 
            if a.check_precond(state, a.args):
                state = a(state, a.args).clauses
        return state
    

    def angelic_search(problem, hierarchy, initialPlan):
        """
	[Figure 11.8] A hierarchical planning algorithm that uses angelic semantics to identify and
	commit to high-level plans that work while avoiding high-level plans that don’t. 
	The predicate MAKING-PROGRESS checks to make sure that we aren’t stuck in an infinite regression
	of refinements. 
	At top level, call ANGELIC -SEARCH with [Act ] as the initialPlan .

        initialPlan contains a sequence of HLA's with angelic semantics 

        The possible effects of an angelic HLA in initialPlan are : 
        ~ : effect remove
        $+: effect possibly add
        $-: effect possibly remove
        $$: possibly add or remove
	"""
        frontier = deque(initialPlan)
        while True: 
            if not frontier:
                return None
            plan = frontier.popleft() # sequence of HLA/Angelic HLA's 
            opt_reachable_set = Problem.reach_opt(problem.init, plan)
            pes_reachable_set = Problem.reach_pes(problem.init, plan)
            if problem.intersects_goal(opt_reachable_set): 
                if Problem.is_primitive( plan, hierarchy ): 
                    return ([x for x in plan.action])
                guaranteed = problem.intersects_goal(pes_reachable_set) 
                if guaranteed and Problem.making_progress(plan, initialPlan):
                    final_state = guaranteed[0] # any element of guaranteed 
                    #print('decompose')
                    return Problem.decompose(hierarchy, problem, plan, final_state, pes_reachable_set)
                (hla, index) = Problem.find_hla(plan, hierarchy) # there should be at least one HLA/Angelic_HLA, otherwise plan would be primitive.
                prefix = plan.action[:index]
                suffix = plan.action[index+1:]
                outcome = Problem(Problem.result(problem.init, prefix), problem.goals , problem.actions )
                for sequence in Problem.refinements(hla, outcome, hierarchy): # find refinements
                    frontier.append(Angelic_Node(outcome.init, plan, prefix + sequence+ suffix, prefix+sequence+suffix))


    def intersects_goal(problem, reachable_set):
        """
        Find the intersection of the reachable states and the goal
        """
        return [y for x in list(reachable_set.keys()) for y in reachable_set[x] if all(goal in y for goal in problem.goals)] 


    def is_primitive(plan,  library):
        """
        checks if the hla is primitive action 
        """
        for hla in plan.action: 
            indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
            for i in indices:
                if library["steps"][i]: 
                    return False
        return True
             


    def reach_opt(init, plan): 
        """
        Finds the optimistic reachable set of the sequence of actions in plan 
        """
        reachable_set = {0: [init]}
        optimistic_description = plan.action #list of angelic actions with optimistic description
        return Problem.find_reachable_set(reachable_set, optimistic_description)
 

    def reach_pes(init, plan): 
        """ 
        Finds the pessimistic reachable set of the sequence of actions in plan
        """
        reachable_set = {0: [init]}
        pessimistic_description = plan.action_pes # list of angelic actions with pessimistic description
        return Problem.find_reachable_set(reachable_set, pessimistic_description)

    def find_reachable_set(reachable_set, action_description):
        """
	Finds the reachable states of the action_description when applied in each state of reachable set.
	"""
        for i in range(len(action_description)):
            reachable_set[i+1]=[]
            if type(action_description[i]) is Angelic_HLA:
                possible_actions = action_description[i].angelic_action()
            else: 
                possible_actions = action_description
            for action in possible_actions:
                for state in reachable_set[i]:
                    if action.check_precond(state , action.args) :
                        if action.effect[0] :
                            new_state = action(state, action.args).clauses
                            reachable_set[i+1].append(new_state)
                        else: 
                            reachable_set[i+1].append(state)
        return reachable_set

    def find_hla(plan, hierarchy):
        """
        Finds the the first HLA action in plan.action, which is not primitive
        and its corresponding index in plan.action
        """
        hla = None
        index = len(plan.action)
        for i in range(len(plan.action)): # find the first HLA in plan, that is not primitive
            if not Problem.is_primitive(Node(plan.state, plan.parent, [plan.action[i]]), hierarchy):
                hla = plan.action[i] 
                index = i
                break
        return (hla, index)
	
    def making_progress(plan, initialPlan):
        """ 
        Prevents from infinite regression of refinements  

        (infinite regression of refinements happens when the algorithm finds a plan that 
        its pessimistic reachable set intersects the goal inside a call to decompose on the same plan, in the same circumstances)  
        """
        for i in range(len(initialPlan)):
            if (plan == initialPlan[i]):
                return False
        return True 

    def decompose(hierarchy, s_0, plan, s_f, reachable_set):
        solution = [] 
        i = max(reachable_set.keys())
        while plan.action_pes: 
            action = plan.action_pes.pop()
            if (i==0): 
                return solution
            s_i = Problem.find_previous_state(s_f, reachable_set,i, action) 
            problem = Problem(s_i, s_f , plan.action)
            angelic_call = Problem.angelic_search(problem, hierarchy, [Angelic_Node(s_i, Node(None), [action],[action])])
            if angelic_call:
                for x in angelic_call: 
                    solution.insert(0,x)
            else: 
                return None
            s_f = s_i
            i-=1
        return solution


    def find_previous_state(s_f, reachable_set, i, action):
        """
        Given a final state s_f and an action finds a state s_i in reachable_set 
        such that when action is applied to state s_i returns s_f.  
        """
        s_i = reachable_set[i-1][0]
        for state in reachable_set[i-1]:
            if s_f in [x for x in Problem.reach_pes(state, Angelic_Node(state, None, [action],[action]))[1]]:
                s_i =state
                break
        return s_i


def job_shop_problem():
    """
    [Figure 11.1] JOB-SHOP-PROBLEM

    A job-shop scheduling problem for assembling two cars,
    with resource and ordering constraints.

    Example:
    >>> from planning import *
    >>> p = job_shop_problem()
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[1][0])
    >>> p.act(p.jobs[1][1])
    >>> p.act(p.jobs[1][2])
    >>> p.act(p.jobs[0][0])
    >>> p.act(p.jobs[0][1])
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[0][2])
    >>> p.goal_test()
    True
    >>>
    """
    resources = {'EngineHoists': 1, 'WheelStations': 2, 'Inspectors': 2, 'LugNuts': 500}

    add_engine1 = HLA('AddEngine1', precond='~Has(C1, E1)', effect='Has(C1, E1)', duration=30, use={'EngineHoists': 1})
    add_engine2 = HLA('AddEngine2', precond='~Has(C2, E2)', effect='Has(C2, E2)', duration=60, use={'EngineHoists': 1})
    add_wheels1 = HLA('AddWheels1', precond='~Has(C1, W1)', effect='Has(C1, W1)', duration=30, use={'WheelStations': 1}, consume={'LugNuts': 20})
    add_wheels2 = HLA('AddWheels2', precond='~Has(C2, W2)', effect='Has(C2, W2)', duration=15, use={'WheelStations': 1}, consume={'LugNuts': 20})
    inspect1 = HLA('Inspect1', precond='~Inspected(C1)', effect='Inspected(C1)', duration=10, use={'Inspectors': 1})
    inspect2 = HLA('Inspect2', precond='~Inspected(C2)', effect='Inspected(C2)', duration=10, use={'Inspectors': 1})

    actions = [add_engine1, add_engine2, add_wheels1, add_wheels2, inspect1, inspect2]

    job_group1 = [add_engine1, add_wheels1, inspect1]
    job_group2 = [add_engine2, add_wheels2, inspect2]

    return Problem(init='Car(C1) & Car(C2) & Wheels(W1) & Wheels(W2) & Engine(E2) & Engine(E2) & ~Has(C1, E1) & ~Has(C2, E2) & ~Has(C1, W1) & ~Has(C2, W2) & ~Inspected(C1) & ~Inspected(C2)',
                   goals='Has(C1, W1) & Has(C1, E1) & Inspected(C1) & Has(C2, W2) & Has(C2, E2) & Inspected(C2)',
                   actions=actions,
                   jobs=[job_group1, job_group2],
                   resources=resources)


def go_to_sfo():
    """Go to SFO Problem"""

    go_home_sfo1 = HLA('Go(Home, SFO)', precond='At(Home) & Have(Car)', effect='At(SFO) & ~At(Home)')
    go_home_sfo2 = HLA('Go(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
    drive_home_sfoltp = HLA('Drive(Home, SFOLongTermParking)', precond='At(Home) & Have(Car)', effect='At(SFOLongTermParking) & ~At(Home)')
    shuttle_sfoltp_sfo = HLA('Shuttle(SFOLongTermParking, SFO)', precond='At(SFOLongTermParking)', effect='At(SFO) & ~At(SFOLongTermParking)')
    taxi_home_sfo = HLA('Taxi(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')

    actions = [go_home_sfo1, go_home_sfo2, drive_home_sfoltp, shuttle_sfoltp_sfo, taxi_home_sfo]

    library = {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
        ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
        ],
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
        ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
        ]
    }

    return Problem(init='At(Home)', goals='At(SFO)', actions=actions), library


class Angelic_HLA(HLA):
    """
    Define Actions for the real-world (that may be refined further), under angelic semantics
    """
    
    def __init__(self, action, precond , effect, duration =0, consume = None, use = None):
        super().__init__(action, precond, effect, duration, consume, use)


    def convert(self, clauses):
        """
        Converts strings into Exprs
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable ) 
        and furthermore can have following effects on the variables: 
            Possibly add variable    ( $+ )
            Possibly remove variable ( $- )
            Possibly add or remove a variable ( $$ )

        Overrides HLA.convert function
        """ 
        lib = {'~': 'Not', 
                '$+': 'PosYes',
               '$-': 'PosNot',
               '$$' : 'PosYesNot'}

        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                for ch in lib.keys():
                    if clauses[i].op == ch:
                        clauses[i] = expr( lib[ch] + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            for ch in lib.keys():
                clauses = clauses.replace(ch, lib[ch])
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

        
        

    def angelic_action(self):
        """
        Converts a high level action (HLA) with angelic semantics into all of its corresponding high level actions (HLA). 
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable) 
        and furthermore can have following effects for each variable: 

            Possibly add variable ( $+: 'PosYes' )        --> corresponds to two HLAs: 
                                                                HLA_1: add variable  
                                                                HLA_2: leave variable unchanged

            Possibly remove variable ( $-: 'PosNot' )     --> corresponds to two HLAs:
                                                                HLA_1: remove variable
                                                                HLA_2: leave variable unchanged

            Possibly add / remove a variable ( $$: 'PosYesNot' )  --> corresponds to three HLAs: 
                                                                        HLA_1: add variable
                                                                        HLA_2: remove variable
                                                                        HLA_3: leave variable unchanged 


            example: the angelic action with effects possibly add A and possibly add or remove B corresponds to the following 6 effects of HLAs:
            
            
            '$+A & $$B':    HLA_1: 'A & B'   (add A and add B)
                            HLA_2: 'A & ~B'  (add A and remove B)
                            HLA_3: 'A'       (add A)
                            HLA_4: 'B'       (add B)
                            HLA_5: '~B'      (remove B)
                            HLA_6: ' '       (no effect)  

        """

        effects=[[]]
        for clause in self.effect:
            (n,w) = Angelic_HLA.compute_parameters(clause, effects)
            effects = effects*n # create n copies of effects 
            it=range(1)
            if len(effects)!=0:
                # split effects into n sublists (seperate n copies created in compute_parameters)
                it = range(len(effects)//n)
            for i in it:
                if effects[i]:
                    if clause.args: 
                        effects[i] = expr(str(effects[i]) + '&' + str(Expr(clause.op[w:],clause.args[0]))) # make changes in the ith part of effects
                        if n==3:
                            effects[i+len(effects)//3]= expr(str(effects[i+len(effects)//3]) + '&' + str(Expr(clause.op[6:],clause.args[0])))
                    else: 
                        effects[i] = expr(str(effects[i]) + '&' + str(expr(clause.op[w:]))) # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = expr(str(effects[i+len(effects)//3]) + '&' + str(expr(clause.op[6:])))

                else: 
                    if clause.args: 
                        effects[i] = Expr(clause.op[w:], clause.args[0]) # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = Expr(clause.op[6:], clause.args[0])

                    else: 
                        effects[i] = expr(clause.op[w:])  # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = expr(clause.op[6:])
            #print('effects',  effects)

        return [ HLA(Expr(self.name, self.args), self.precond, effects[i] ) for i in range(len(effects)) ]


    def compute_parameters(clause, effects):
        """ 
        computes n,w 
        
        n = number of HLA effects that the anelic HLA corresponds to
        w = length of representation of angelic HLA effect 

                    n = 1, if effect is add
                    n = 1, if effect is remove
                    n = 2, if effect is possibly add
                    n = 2, if effect is possibly remove
                    n = 3, if effect is possibly add or remove

        """
        if clause.op[:9] == 'PosYesNot':
            # possibly add/remove variable: three possible effects for the variable 
            n=3
            w=9
        elif clause.op[:6] == 'PosYes': # possibly add variable: two possible effects for the variable
            n=2
            w=6
        elif clause.op[:6] == 'PosNot': # possibly remove variable: two possible effects for the variable
            n=2
            w=3 # We want to keep 'Not' from 'PosNot' when adding action
        else:   # variable or ~variable 
            n=1
            w=0
        return (n,w)


class Angelic_Node(Node):
    """ 
    Extends the class Node. 
    self.action:     contains the optimistic description of an angelic HLA
    self.action_pes: contains the pessimistic description of an angelic HLA
    """

    def __init__(self, state, parent=None, action_opt=None, action_pes=None,  path_cost=0):
        super().__init__(state, parent, action_opt , path_cost)
        self.action_pes = action_pes 


