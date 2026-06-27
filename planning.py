"""Planning (Chapters 10-11)"""

import copy
import itertools
from collections import deque, defaultdict
from functools import reduce as _reduce

import numpy as np

import search
from csp import sat_up, NaryCSP, Constraint, ac_search_solver, is_constraint
from logic import FolKB, conjuncts, unify_mm, associate, SAT_plan, cdcl_satisfiable
from search import Node
from utils import Expr, expr, first


class PlanningProblem:
    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial, goals, actions, domain=None):
        self.initial = self.convert(initial) if domain is None else self.convert(initial) + self.convert(domain)
        self.goals = self.convert(goals)
        self.actions = actions
        self.domain = domain

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
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def expand_fluents(self, name=None):
        """
        Generate all ground fluents obtainable by binding the problem's objects to the
        predicates appearing in the initial state, goals and action effects. If a fluent
        ``name`` is given, only expansions of that single fluent are produced. When a domain
        is defined, candidate ground fluents are filtered through a FolKB so that only those
        consistent with the domain are kept.
        """

        kb = None
        if self.domain:
            kb = FolKB(self.convert(self.domain))
            for action in self.actions:
                if action.precond:
                    for fests in set(action.precond).union(action.effect).difference(self.convert(action.domain)):
                        if fests.op[:3] != 'Not':
                            kb.tell(expr(str(action.domain) + ' ==> ' + str(fests)))

        objects = set(arg for clause in set(self.initial + self.goals) for arg in clause.args)
        fluent_list = []
        if name is not None:
            for fluent in self.initial + self.goals:
                if str(fluent) == name:
                    fluent_list.append(fluent)
                    break
        else:
            fluent_list = list(map(lambda fluent: Expr(fluent[0], *fluent[1]),
                                   {fluent.op: fluent.args for fluent in self.initial + self.goals +
                                    [clause for action in self.actions for clause in action.effect if
                                     clause.op[:3] != 'Not']}.items()))

        expansions = []
        for fluent in fluent_list:
            for permutation in itertools.permutations(objects, len(fluent.args)):
                new_fluent = Expr(fluent.op, *permutation)
                if (self.domain and kb.ask(new_fluent) is not False) or not self.domain:
                    expansions.append(new_fluent)

        return expansions

    def expand_actions(self, name=None):
        """Generate all possible actions with variable bindings for precondition selection heuristic"""

        has_domains = all(action.domain for action in self.actions if action.precond)
        kb = None
        if has_domains:
            kb = FolKB(self.initial)
            for action in self.actions:
                if action.precond:
                    kb.tell(expr(str(action.domain) + ' ==> ' + str(action)))

        objects = set(arg for clause in self.initial for arg in clause.args)
        expansions = []
        action_list = []
        if name is not None:
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)
                    break
        else:
            action_list = self.actions

        for action in action_list:
            for permutation in itertools.permutations(objects, len(action.args)):
                bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                if bindings is not None:
                    new_args = []
                    for arg in action.args:
                        if arg in bindings:
                            new_args.append(bindings[arg])
                        else:
                            new_args.append(arg)
                    new_expr = Expr(str(action.name), *new_args)
                    if (has_domains and kb.ask(new_expr) is not False) or (
                            has_domains and not action.precond) or not has_domains:
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

    def is_strips(self):
        """
        Returns True if the problem does not contain negative literals in preconditions and goals
        """
        return (all(clause.op[:3] != 'Not' for clause in self.goals) and
                all(clause.op[:3] != 'Not' for action in self.actions for clause in action.precond))

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.initial for goal in self.goals)

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
        if not list_action.check_precond(self.initial, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.initial = list_action(self.initial, args).clauses


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

    def __init__(self, action, precond, effect, domain=None):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond) if domain is None else self.convert(precond) + self.convert(domain)
        self.effect = self.convert(effect)
        self.domain = domain

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}'.format(Expr(self.name, *self.args))

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

    def relaxed(self):
        """
        Removes delete list from the action by removing all negative literals from action's effect
        """
        return Action(Expr(self.name, *self.args), self.precond,
                      list(filter(lambda effect: effect.op[:3] != 'Not', self.effect)))

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

    return PlanningProblem(initial='At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK)',
                           goals='At(C1, JFK) & At(C2, SFO)',
                           actions=[Action('Load(c, p, a)',
                                           precond='At(c, a) & At(p, a)',
                                           effect='In(c, p) & ~At(c, a)',
                                           domain='Cargo(c) & Plane(p) & Airport(a)'),
                                    Action('Unload(c, p, a)',
                                           precond='In(c, p) & At(p, a)',
                                           effect='At(c, a) & ~In(c, p)',
                                           domain='Cargo(c) & Plane(p) & Airport(a)'),
                                    Action('Fly(p, f, to)',
                                           precond='At(p, f)',
                                           effect='At(p, to) & ~At(p, f)',
                                           domain='Plane(p) & Airport(f) & Airport(to)')],
                           domain='Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)')


def spare_tire():
    """
    [Figure 10.2] SPARE-TIRE-PROBLEM

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

    return PlanningProblem(initial='At(Flat, Axle) & At(Spare, Trunk)',
                           goals='At(Spare, Axle) & At(Flat, Ground)',
                           actions=[Action('Remove(obj, loc)',
                                           precond='At(obj, loc)',
                                           effect='At(obj, Ground) & ~At(obj, loc)',
                                           domain='Tire(obj)'),
                                    Action('PutOn(t, Axle)',
                                           precond='At(t, Ground) & ~At(Flat, Axle)',
                                           effect='At(t, Axle) & ~At(t, Ground)',
                                           domain='Tire(t)'),
                                    Action('LeaveOvernight',
                                           precond='',
                                           effect='~At(Spare, Ground) & ~At(Spare, Axle) & ~At(Spare, Trunk) & \
                                        ~At(Flat, Ground) & ~At(Flat, Axle) & ~At(Flat, Trunk)')],
                           domain='Tire(Flat) & Tire(Spare)')


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
    return PlanningProblem(initial='On(A, Table) & On(B, Table) & On(C, A) & Clear(B) & Clear(C)',
                           goals='On(A, B) & On(B, C)',
                           actions=[Action('Move(b, x, y)',
                                           precond='On(b, x) & Clear(b) & Clear(y)',
                                           effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)',
                                           domain='Block(b) & Block(y)'),
                                    Action('MoveToTable(b, x)',
                                           precond='On(b, x) & Clear(b)',
                                           effect='On(b, Table) & Clear(x) & ~On(b, x)',
                                           domain='Block(b) & Block(x)')],
                           domain='Block(A) & Block(B) & Block(C)')


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

    return PlanningProblem(initial='On(A, B) & Clear(A) & OnTable(B) & OnTable(C) & Clear(C)',
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

    return PlanningProblem(initial='Have(Cake)',
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

    return PlanningProblem(initial='At(Home) & Sells(SM, Milk) & Sells(SM, Banana) & Sells(HW, Drill)',
                           goals='Have(Milk) & Have(Banana) & Have(Drill)',
                           actions=[Action('Buy(x, store)',
                                           precond='At(store) & Sells(store, x)',
                                           effect='Have(x)',
                                           domain='Store(store) & Item(x)'),
                                    Action('Go(x, y)',
                                           precond='At(x)',
                                           effect='At(y) & ~At(x)',
                                           domain='Place(x) & Place(y)')],
                           domain='Place(Home) & Place(SM) & Place(HW) & Store(SM) & Store(HW) & '
                                  'Item(Milk) & Item(Banana) & Item(Drill)')


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

    return PlanningProblem(initial='',
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


def logistics_problem(initial_state=None, goal_state=None):
    """
    LOGISTICS-PROBLEM

    A logistics problem where a robot moves between places, picking up and
    putting down containers in order to deliver them to their destinations.

    Example:
    >>> from planning import *
    >>> lp = logistics_problem(goal_state='In(C2, D3) & In(C3, D3)')
    >>> lp.goal_test()
    False
    >>> lp.act(expr('PutDown(R1, C1, D1)'))
    >>> lp.act(expr('PickUp(R1, C2, D1)'))
    >>> lp.act(expr('Move(R1, D1, D3)'))
    >>> lp.act(expr('PutDown(R1, C2, D3)'))
    >>> lp.act(expr('Move(R1, D3, D2)'))
    >>> lp.act(expr('PickUp(R1, C3, D2)'))
    >>> lp.act(expr('Move(R1, D2, D3)'))
    >>> lp.goal_test()
    False
    >>> lp.act(expr('PutDown(R1, C3, D3)'))
    >>> lp.goal_test()
    True
    >>>
    """
    if initial_state is None:
        initial_state = 'In(C1, R1) & In(C2, D1) & In(C3, D2) & In(R1, D1) & Holding(R1)'
    if goal_state is None:
        raise ValueError('Goal must be defined')

    return PlanningProblem(initial=initial_state,
                           goals=goal_state,
                           actions=[Action('PickUp(r, c, d)',
                                           precond='In(r, d) & In(c, d) & ~Holding(r)',
                                           effect='Holding(r) & ~In(c, d) & In(c, r)',
                                           domain='Robot(r) & Place(d) & Container(c)'),
                                    Action('PutDown(r, c, d)',
                                           precond='In(r, d) & In(c, r) & Holding(r)',
                                           effect='~Holding(r) & ~In(c, r) & In(c, d)',
                                           domain='Robot(r) & Place(d) & Container(c)'),
                                    Action('Move(r, d_start, d_end)',
                                           precond='In(r, d_start)',
                                           effect='~In(r, d_start) & In(r, d_end)',
                                           domain='Robot(r) & Place(d_start) & Place(d_end)')],
                           domain='Container(C1) & Container(C2) & Container(C3) & '
                                  'Place(D1) & Place(D2) & Place(D3) & Robot(R1)')


def blocks_world(initial, goals, blocks):
    """
    GENERALIZED-BLOCKS-WORLD-PROBLEM

    A flexible constructor for creating blocks-world planning problems.
    Any initial and goal configuration can be specified for a given set of blocks.

    Example:
    >>> from planning import *
    >>> initial_state = 'On(C, A) & On(A, Table) & On(B, Table) & Clear(C) & Clear(B)'
    >>> goal_state = 'On(A, B) & On(B, C)'
    >>> sussman_anomaly = blocks_world(initial_state, goal_state, ['A', 'B', 'C'])
    >>> sussman_anomaly.goal_test()
    False
    >>> sussman_anomaly.act(expr('MoveToTable(C, A)'))
    >>> sussman_anomaly.act(expr('Move(B, Table, C)'))
    >>> sussman_anomaly.act(expr('Move(A, Table, B)'))
    >>> sussman_anomaly.goal_test()
    True
    >>>
    """
    # Dynamically generate the domain knowledge based on the list of blocks
    domain = ' & '.join('Block({})'.format(b) for b in blocks)

    actions = [Action('Move(b, x, y)',
                      precond='On(b, x) & Clear(b) & Clear(y)',
                      effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)',
                      domain='Block(b) & Block(y)'),
               Action('MoveToTable(b, x)',
                      precond='On(b, x) & Clear(b)',
                      effect='On(b, Table) & Clear(x) & ~On(b, x)',
                      domain='Block(b) & Block(x)')]

    return PlanningProblem(initial=initial,
                           goals=goals,
                           actions=actions,
                           domain=domain)


def rush_hour():
    """
    RUSH-HOUR-PROBLEM (non-numeric version)

    A planning problem for the Rush Hour sliding block puzzle. The goal is to
    maneuver the RedCar to the exit. This version uses non-numeric symbols for
    grid positions (e.g. R1, C1) instead of integers.

    This specific instance uses:
    - RedCar (2x1, horizontal) starting at (R3, C1)
    - GreenTruck (3x1, vertical) starting at (R1, C4)
    - BlueCar (2x1, vertical) starting at (R5, C2)
    """
    initial_state = 'At(RedCar, R3, C1) & At(GreenTruck, R1, C4) & At(BlueCar, R5, C2) & ' \
                    'IsHorizontal(RedCar) & IsVertical(GreenTruck) & IsVertical(BlueCar) & ' \
                    'Clear(R1, C1) & Clear(R1, C2) & Clear(R1, C3) & Clear(R1, C5) & Clear(R1, C6) & ' \
                    'Clear(R2, C1) & Clear(R2, C2) & Clear(R2, C3) & Clear(R2, C5) & Clear(R2, C6) & ' \
                    'Clear(R3, C3) & Clear(R3, C4) & Clear(R3, C5) & Clear(R3, C6) & ' \
                    'Clear(R4, C1) & Clear(R4, C2) & Clear(R4, C3) & Clear(R4, C4) & Clear(R4, C5) & Clear(R4, C6) & ' \
                    'Clear(R5, C1) & Clear(R5, C3) & Clear(R5, C4) & Clear(R5, C5) & Clear(R5, C6) & ' \
                    'Clear(R6, C1) & Clear(R6, C3) & Clear(R6, C4) & Clear(R6, C5) & Clear(R6, C6)'

    # Goal state: the RedCar's left-most part is at column C5.
    goal_state = 'At(RedCar, R3, C5)'

    domain = 'Vehicle(RedCar) & Vehicle(GreenTruck) & Vehicle(BlueCar) & ' \
             'Car(RedCar) & Truck(GreenTruck) & Car(BlueCar) & ' \
             'Row(R1) & Row(R2) & Row(R3) & Row(R4) & Row(R5) & Row(R6) & ' \
             'Col(C1) & Col(C2) & Col(C3) & Col(C4) & Col(C5) & Col(C6) & ' \
             'NextTo(R1, R2) & NextTo(R2, R3) & NextTo(R3, R4) & NextTo(R4, R5) & NextTo(R5, R6) & ' \
             'NextTo(C1, C2) & NextTo(C2, C3) & NextTo(C3, C4) & NextTo(C4, C5) & NextTo(C5, C6)'

    actions = [
        # car actions (length 2)
        Action('MoveRightCar(v, r, c1, c2, c3)',
               precond='At(v, r, c1) & Car(v) & IsHorizontal(v) & NextTo(c1, c2) & NextTo(c2, c3) & Clear(r, c3)',
               effect='At(v, r, c2) & ~At(v, r, c1) & Clear(r, c1) & ~Clear(r, c3)',
               domain='Vehicle(v) & Row(r) & Col(c1) & Col(c2) & Col(c3)'),
        Action('MoveLeftCar(v, r, c1, c2, c3)',
               precond='At(v, r, c2) & Car(v) & IsHorizontal(v) & NextTo(c1, c2) & NextTo(c2, c3) & Clear(r, c1)',
               effect='At(v, r, c1) & ~At(v, r, c2) & Clear(r, c3) & ~Clear(r, c1)',
               domain='Vehicle(v) & Row(r) & Col(c1) & Col(c2) & Col(c3)'),
        Action('MoveDownCar(v, r1, r2, r3, c)',
               precond='At(v, r1, c) & Car(v) & IsVertical(v) & NextTo(r1, r2) & NextTo(r2, r3) & Clear(r3, c)',
               effect='At(v, r2, c) & ~At(v, r1, c) & Clear(r1, c) & ~Clear(r3, c)',
               domain='Vehicle(v) & Row(r1) & Row(r2) & Row(r3) & Col(c)'),
        Action('MoveUpCar(v, r1, r2, r3, c)',
               precond='At(v, r2, c) & Car(v) & IsVertical(v) & NextTo(r1, r2) & NextTo(r2, r3) & Clear(r1, c)',
               effect='At(v, r1, c) & ~At(v, r2, c) & Clear(r3, c) & ~Clear(r1, c)',
               domain='Vehicle(v) & Row(r1) & Row(r2) & Row(r3) & Col(c)'),
        # truck actions (length 3)
        Action('MoveRightTruck(v, r, c1, c2, c3, c4)',
               precond='At(v, r, c1) & Truck(v) & IsHorizontal(v) & NextTo(c1, c2) & NextTo(c2, c3) & '
                       'NextTo(c3, c4) & Clear(r, c4)',
               effect='At(v, r, c2) & ~At(v, r, c1) & Clear(r, c1) & ~Clear(r, c4)',
               domain='Vehicle(v) & Row(r) & Col(c1) & Col(c2) & Col(c3) & Col(c4)'),
        Action('MoveLeftTruck(v, r, c1, c2, c3, c4)',
               precond='At(v, r, c2) & Truck(v) & IsHorizontal(v) & NextTo(c1, c2) & NextTo(c2, c3) & '
                       'NextTo(c3, c4) & Clear(r, c1)',
               effect='At(v, r, c1) & ~At(v, r, c2) & Clear(r, c4) & ~Clear(r, c1)',
               domain='Vehicle(v) & Row(r) & Col(c1) & Col(c2) & Col(c3) & Col(c4)'),
        Action('MoveDownTruck(v, r1, r2, r3, r4, c)',
               precond='At(v, r1, c) & Truck(v) & IsVertical(v) & NextTo(r1, r2) & NextTo(r2, r3) & '
                       'NextTo(r3, r4) & Clear(r4, c)',
               effect='At(v, r2, c) & ~At(v, r1, c) & Clear(r1, c) & ~Clear(r4, c)',
               domain='Vehicle(v) & Row(r1) & Row(r2) & Row(r3) & Row(r4) & Col(c)'),
        Action('MoveUpTruck(v, r1, r2, r3, r4, c)',
               precond='At(v, r2, c) & Truck(v) & IsVertical(v) & NextTo(r1, r2) & NextTo(r2, r3) & '
                       'NextTo(r3, r4) & Clear(r1, c)',
               effect='At(v, r1, c) & ~At(v, r2, c) & Clear(r4, c) & ~Clear(r1, c)',
               domain='Vehicle(v) & Row(r1) & Row(r2) & Row(r3) & Row(r4) & Col(c)')]

    return PlanningProblem(initial=initial_state,
                           goals=goal_state,
                           actions=actions,
                           domain=domain)


def rush_hour_optimized():
    """
    RUSH-HOUR-PROBLEM (optimized version)

    This version optimizes the planning problem by creating vehicle-specific
    actions. Since each vehicle's orientation is fixed, generic predicates like
    IsHorizontal can be removed and actions can be created that only apply to the
    correct vehicle on its fixed axis of movement. This drastically reduces the
    number of permutations the planner needs to generate and check.
    """
    # Initial state is simpler as orientation is now baked into the actions.
    initial_state = 'At(RedCar, R3, C1) & At(GreenTruck, R1, C4) & At(BlueCar, R5, C2) & ' \
                    'Clear(R1, C1) & Clear(R1, C2) & Clear(R1, C3) & Clear(R1, C5) & Clear(R1, C6) & ' \
                    'Clear(R2, C1) & Clear(R2, C2) & Clear(R2, C3) & Clear(R2, C5) & Clear(R2, C6) & ' \
                    'Clear(R3, C3) & Clear(R3, C4) & Clear(R3, C5) & Clear(R3, C6) & ' \
                    'Clear(R4, C1) & Clear(R4, C2) & Clear(R4, C3) & Clear(R4, C4) & Clear(R4, C5) & Clear(R4, C6) & ' \
                    'Clear(R5, C1) & Clear(R5, C3) & Clear(R5, C4) & Clear(R5, C5) & Clear(R5, C6) & ' \
                    'Clear(R6, C1) & Clear(R6, C3) & Clear(R6, C4) & Clear(R6, C5) & Clear(R6, C6)'

    goal_state = 'At(RedCar, R3, C5)'

    domain = 'Vehicle(RedCar) & Vehicle(GreenTruck) & Vehicle(BlueCar) & ' \
             'Row(R1) & Row(R2) & Row(R3) & Row(R4) & Row(R5) & Row(R6) & ' \
             'Col(C1) & Col(C2) & Col(C3) & Col(C4) & Col(C5) & Col(C6) & ' \
             'NextTo(R1, R2) & NextTo(R2, R3) & NextTo(R3, R4) & NextTo(R4, R5) & NextTo(R5, R6) & ' \
             'NextTo(C1, C2) & NextTo(C2, C3) & NextTo(C3, C4) & NextTo(C4, C5) & NextTo(C5, C6)'

    actions = [
        # RedCar is horizontal on row R3, length 2
        Action('MoveRedCarRight(c1, c2, c3)',
               precond='At(RedCar, R3, c1) & NextTo(c1, c2) & NextTo(c2, c3) & Clear(R3, c3)',
               effect='At(RedCar, R3, c2) & ~At(RedCar, R3, c1) & Clear(R3, c1) & ~Clear(R3, c3)',
               domain='Col(c1) & Col(c2) & Col(c3)'),
        Action('MoveRedCarLeft(c1, c2, c3)',
               precond='At(RedCar, R3, c2) & NextTo(c1, c2) & NextTo(c2, c3) & Clear(R3, c1)',
               effect='At(RedCar, R3, c1) & ~At(RedCar, R3, c2) & Clear(R3, c3) & ~Clear(R3, c1)',
               domain='Col(c1) & Col(c2) & Col(c3)'),
        # GreenTruck is vertical on column C4, length 3
        Action('MoveGreenTruckDown(r1, r2, r3, r4)',
               precond='At(GreenTruck, r1, C4) & NextTo(r1, r2) & NextTo(r2, r3) & NextTo(r3, r4) & Clear(r4, C4)',
               effect='At(GreenTruck, r2, C4) & ~At(GreenTruck, r1, C4) & Clear(r1, C4) & ~Clear(r4, C4)',
               domain='Row(r1) & Row(r2) & Row(r3) & Row(r4)'),
        Action('MoveGreenTruckUp(r1, r2, r3, r4)',
               precond='At(GreenTruck, r2, C4) & NextTo(r1, r2) & NextTo(r2, r3) & NextTo(r3, r4) & Clear(r1, C4)',
               effect='At(GreenTruck, r1, C4) & ~At(GreenTruck, r2, C4) & Clear(r4, C4) & ~Clear(r1, C4)',
               domain='Row(r1) & Row(r2) & Row(r3) & Row(r4)'),
        # BlueCar is vertical on column C2, length 2
        Action('MoveBlueCarDown(r1, r2, r3)',
               precond='At(BlueCar, r1, C2) & NextTo(r1, r2) & NextTo(r2, r3) & Clear(r3, C2)',
               effect='At(BlueCar, r2, C2) & ~At(BlueCar, r1, C2) & Clear(r1, C2) & ~Clear(r3, C2)',
               domain='Row(r1) & Row(r2) & Row(r3)'),
        Action('MoveBlueCarUp(r1, r2, r3)',
               precond='At(BlueCar, r2, C2) & NextTo(r1, r2) & NextTo(r2, r3) & Clear(r1, C2)',
               effect='At(BlueCar, r1, C2) & ~At(BlueCar, r2, C2) & Clear(r3, C2) & ~Clear(r1, C2)',
               domain='Row(r1) & Row(r2) & Row(r3)')]

    return PlanningProblem(initial=initial_state,
                           goals=goal_state,
                           actions=actions,
                           domain=domain)


def double_tennis_problem():
    """
    [Figure 11.10] DOUBLE-TENNIS-PROBLEM

    A multiagent planning problem involving two partner tennis players
    trying to return an approaching ball and repositioning around in the court.

    Example:
    >>> from planning import *
    >>> dtp = double_tennis_problem()
    >>> goal_test(dtp.goals, dtp.initial)
    False
    >>> dtp.act(expr('Go(A, RightBaseLine, LeftBaseLine)'))
    >>> dtp.act(expr('Hit(A, Ball, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.initial)
    False
    >>> dtp.act(expr('Go(A, LeftNet, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.initial)
    True
    >>>
    """

    return PlanningProblem(
        initial='At(A, LeftBaseLine) & At(B, RightNet) & Approaching(Ball, RightBaseLine) & Partner(A, B) & Partner(B, A)',
        goals='Returned(Ball) & At(a, LeftNet) & At(a, RightNet)',
        actions=[Action('Hit(actor, Ball, loc)',
                        precond='Approaching(Ball, loc) & At(actor, loc)',
                        effect='Returned(Ball)'),
                 Action('Go(actor, to, loc)',
                        precond='At(actor, loc)',
                        effect='At(actor, to) & ~At(actor, loc)')])


class ForwardPlan(search.Problem):
    """
    [Section 10.2.1]
    Forward state-space search
    """

    def __init__(self, planning_problem):
        super().__init__(associate('&', planning_problem.initial), associate('&', planning_problem.goals))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions()

    def actions(self, state):
        """Return the expanded actions whose preconditions all hold in the given state."""
        return [action for action in self.expanded_actions if all(pre in conjuncts(state) for pre in action.precond)]

    def result(self, state, action):
        """Return the state resulting from applying the action to the given state."""
        return associate('&', action(conjuncts(state), action.args).clauses)

    def goal_test(self, state):
        """Return True if every goal of the planning problem holds in the given state."""
        return all(goal in conjuncts(state) for goal in self.planning_problem.goals)

    def h(self, state):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=state.state,
                                                   goals=self.goal,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf


class BackwardPlan(search.Problem):
    """
    [Section 10.2.2]
    Backward relevant-states search
    """

    def __init__(self, planning_problem):
        super().__init__(associate('&', planning_problem.goals), associate('&', planning_problem.initial))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions()

    def actions(self, subgoal):
        """
        Returns True if the action is relevant to the subgoal, i.e.:
        - the action achieves an element of the effects
        - the action doesn't delete something that needs to be achieved
        - the preconditions are consistent with other subgoals that need to be achieved
        """

        def negate_clause(clause):
            return Expr(clause.op.replace('Not', ''), *clause.args) if clause.op[:3] == 'Not' else Expr(
                'Not' + clause.op, *clause.args)

        subgoal = conjuncts(subgoal)
        return [action for action in self.expanded_actions if
                (any(prop in action.effect for prop in subgoal) and
                 not any(negate_clause(prop) in subgoal for prop in action.effect) and
                 not any(negate_clause(prop) in subgoal and negate_clause(prop) not in action.effect
                         for prop in action.precond))]

    def result(self, subgoal, action):
        """Regress the subgoal through the action, i.e. ``g' = (g - effects(a)) + preconds(a)``."""
        # g' = (g - effects(a)) + preconds(a)
        return associate('&', set(set(conjuncts(subgoal)).difference(action.effect)).union(action.precond))

    def goal_test(self, subgoal):
        """Return True if the subgoal is entailed by the search goal (the problem's initial state)."""
        return all(goal in conjuncts(self.goal) for goal in conjuncts(subgoal))

    def h(self, subgoal):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=self.goal,
                                                   goals=subgoal.state,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf


def CSPlan(planning_problem, solution_length, CSP_solver=ac_search_solver, arc_heuristic=sat_up):
    """
    [Section 10.4.3]
    Planning as Constraint Satisfaction Problem
    """

    def st(var, stage):
        """Returns a string for the var-stage pair that can be used as a variable"""
        return str(var) + "_" + str(stage)

    def if_(v1, v2):
        """If the second argument is v2, the first argument must be v1"""

        def if_fun(x1, x2):
            return x1 == v1 if x2 == v2 else True

        if_fun.__name__ = "if the second argument is " + str(v2) + " then the first argument is " + str(v1) + " "
        return if_fun

    def eq_if_not_in_(actset):
        """First and third arguments are equal if action is not in actset"""

        def eq_if_not_in(x1, a, x2):
            return x1 == x2 if a not in actset else True

        eq_if_not_in.__name__ = "first and third arguments are equal if action is not in " + str(actset) + " "
        return eq_if_not_in

    expanded_actions = planning_problem.expand_actions()
    fluent_values = planning_problem.expand_fluents()
    for horizon in range(solution_length):
        act_vars = [st('action', stage) for stage in range(horizon + 1)]
        domains = {av: list(map(lambda action: expr(str(action)), expanded_actions)) for av in act_vars}
        domains.update({st(var, stage): {True, False} for var in fluent_values for stage in range(horizon + 2)})
        # initial state constraints
        constraints = [Constraint((st(var, 0),), is_constraint(val))
                       for (var, val) in {expr(str(fluent).replace('Not', '')):
                                              True if fluent.op[:3] != 'Not' else False
                                          for fluent in planning_problem.initial}.items()]
        constraints += [Constraint((st(var, 0),), is_constraint(False))
                        for var in {expr(str(fluent).replace('Not', ''))
                                    for fluent in fluent_values if fluent not in planning_problem.initial}]
        # goal state constraints
        constraints += [Constraint((st(var, horizon + 1),), is_constraint(val))
                        for (var, val) in {expr(str(fluent).replace('Not', '')):
                                               True if fluent.op[:3] != 'Not' else False
                                           for fluent in planning_problem.goals}.items()]
        # precondition constraints
        constraints += [Constraint((st(var, stage), st('action', stage)), if_(val, act))
                        # st(var, stage) == val if st('action', stage) == act
                        for act, strps in {expr(str(action)): action for action in expanded_actions}.items()
                        for var, val in {expr(str(fluent).replace('Not', '')):
                                             True if fluent.op[:3] != 'Not' else False
                                         for fluent in strps.precond}.items()
                        for stage in range(horizon + 1)]
        # effect constraints
        constraints += [Constraint((st(var, stage + 1), st('action', stage)), if_(val, act))
                        # st(var, stage + 1) == val if st('action', stage) == act
                        for act, strps in {expr(str(action)): action for action in expanded_actions}.items()
                        for var, val in {expr(str(fluent).replace('Not', '')): True if fluent.op[:3] != 'Not' else False
                                         for fluent in strps.effect}.items()
                        for stage in range(horizon + 1)]
        # frame constraints
        constraints += [Constraint((st(var, stage), st('action', stage), st(var, stage + 1)),
                                   eq_if_not_in_(set(map(lambda action: expr(str(action)),
                                                         {act for act in expanded_actions if var in act.effect
                                                          or Expr('Not' + var.op, *var.args) in act.effect}))))
                        for var in fluent_values for stage in range(horizon + 1)]
        csp = NaryCSP(domains, constraints)
        sol = CSP_solver(csp, arc_heuristic=arc_heuristic)
        if sol:
            return [sol[a] for a in act_vars]


def SATPlan(planning_problem, solution_length, SAT_solver=cdcl_satisfiable):
    """
    [Section 10.4.1]
    Planning as Boolean satisfiability
    """

    def expand_transitions(state, actions):
        state = sorted(conjuncts(state))
        for action in filter(lambda act: act.check_precond(state, act.args), actions):
            transition[associate('&', state)].update(
                {Expr(action.name, *action.args):
                     associate('&', sorted(set(filter(lambda clause: clause.op[:3] != 'Not',
                                                      action(state, action.args).clauses))))
                     if planning_problem.is_strips()
                     else associate('&', sorted(set(action(state, action.args).clauses)))})
        for state in transition[associate('&', state)].values():
            if state not in transition:
                expand_transitions(expr(state), actions)

    transition = defaultdict(dict)
    expand_transitions(associate('&', planning_problem.initial), planning_problem.expand_actions())

    return SAT_plan(associate('&', sorted(planning_problem.initial)), transition,
                    associate('&', sorted(planning_problem.goals)), solution_length, SAT_solver=SAT_solver)


def predicate_negate(e):
    """Return the logical negation of an Expr predicate, avoiding a double 'Not' prefix."""
    return Expr(e.op[3:], *e.args) if e.op.startswith('Not') else Expr('Not' + e.op, *e.args)


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
        # Action -> preconditions for that action
        self.current_action_links = {}
        # current state to action link
        # Precondition -> what is applicable (actions)
        self.current_state_links = {}
        # current action to next state link (e.g. Go(Home, HW) --> At(HW) and NotAt(Home))
        # aka forward link in time (dependency)
        self.next_action_links = {}
        # next state to current action link (e.g. NotAt(Home): [Go(Home, HW), Go(Home, SM)])
        # aka backwards link in time (dependency)
        self.next_state_links = {}
        # mutually exclusive actions
        self.action_mutexes = []
        # mutually exclusive states
        self.state_mutexes = []

    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()

    def __str__(self):
        state_str = ', '.join(str(s) for s in self.current_state)
        action_str = ', '.join(str(a) for a in self.current_action_links.keys())
        mutex_str = ', '.join(str(m) for m in self.action_mutexes)
        return ('<Level>\n'
                '  Current State: {{{}}}\n'
                '  Actions: {{{}}}\n'
                '  Mutex: {{{}}}\n'.format(state_str, action_str, mutex_str))

    __repr__ = __str__

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

        # clear out effects from state mutex prior computation
        self.action_mutexes = []

        # Competing needs - two actions are mutex if any of their preconditions
        # are mutex at the previous state level
        for a1, a2 in itertools.combinations(self.current_action_links.keys(), 2):
            preconds_a1 = self.current_action_links[a1]
            preconds_a2 = self.current_action_links[a2]

            if any({p, q} in self.state_mutexes for p in preconds_a1 for q in preconds_a2):
                mutex_pair = {a1, a2}
                if mutex_pair not in self.action_mutexes:
                    self.action_mutexes.append(mutex_pair)

        # Interference and inconsistent effects mutex calculation
        for a1, a2 in itertools.combinations(self.next_action_links.keys(), 2):
            preconds_a1 = self.current_action_links.get(a1, [])
            preconds_a2 = self.current_action_links.get(a2, [])
            effects_a1 = self.next_action_links.get(a1, [])
            effects_a2 = self.next_action_links.get(a2, [])

            interference = False
            # Interference check
            for p1 in preconds_a1:
                if predicate_negate(p1) in effects_a2:
                    interference = True
            for p2 in preconds_a2:
                if predicate_negate(p2) in effects_a1:
                    interference = True

            # Inconsistent effects check
            for e1 in effects_a1:
                if predicate_negate(e1) in effects_a2:
                    interference = True
            for e2 in effects_a2:
                if predicate_negate(e2) in effects_a1:
                    interference = True

            if interference:
                mutex_pair = {a1, a2}
                if mutex_pair not in self.action_mutexes:
                    self.action_mutexes.append(mutex_pair)

    def populate_prop_mutexes(self):
        """Compute the next level's proposition mutexes based on the current action mutexes"""

        # Inconsistent support - two props cannot be true given competing supporting actions
        state_mutex = []
        next_state_pairs = itertools.combinations(self.next_state_links.keys(), 2)
        for next_state_pair in list(next_state_pairs):
            s1, s2 = list(next_state_pair)
            acts_to_s1 = self.next_state_links.get(s1, [])
            acts_to_s2 = self.next_state_links.get(s2, [])

            # ensure our mutexes only apply to pairs, not single states
            if acts_to_s1 == [] or acts_to_s2 == []:
                continue

            # if any two actions that lead to these states are not mutex,
            # do not add a mutex to these states
            if all({a1, a2} in self.action_mutexes or {a2, a1} in self.action_mutexes
                   for a1 in acts_to_s1 for a2 in acts_to_s2):
                mutex_pair = {s1, s2}
                if mutex_pair not in state_mutex:
                    state_mutex.append(mutex_pair)

        # If there are pairs of propositions that are negations of each other, they must be mutex
        for s1i in range(len(self.current_state)):
            for s2i in range(s1i, len(self.current_state)):
                s1, s2 = self.current_state[s1i], self.current_state[s2i]
                if (repr(s2)[0:3] == 'Not' and repr(s1) == repr(s2)[3:] or
                        repr(s1)[0:3] == 'Not' and repr(s1)[3:] == repr(s2)):
                    mutex_pair = {s1, s2}
                    if mutex_pair not in state_mutex:
                        state_mutex.append(mutex_pair)

        return state_mutex

    def prune_invalid_actions(self):
        """Remove actions whose own preconditions are mutex (unsupportable)"""

        to_remove = []

        # Normalize state mutex set for fast membership checks
        state_mutex_lookup = set()
        for m in self.state_mutexes:
            state_mutex_lookup.add(frozenset(m))

        for action, preconds in list(self.current_action_links.items()):
            invalid = False
            for p1, p2 in itertools.combinations(preconds, 2):
                if frozenset({p1, p2}) in state_mutex_lookup:
                    invalid = True
                    break
            if invalid:
                to_remove.append(action)

        # Remove invalid actions from all mappings
        for action in to_remove:
            # forward mappings
            self.current_action_links.pop(action, None)
            self.next_action_links.pop(action, None)

            # reverse mapping: state -> actions (current_state_links)
            for precond in list(self.current_state_links.keys()):
                actions_for_pre = self.current_state_links.get(precond, [])
                if action in actions_for_pre:
                    actions_for_pre.remove(action)
                    if not actions_for_pre:
                        self.current_state_links.pop(precond, None)
                    else:
                        self.current_state_links[precond] = actions_for_pre

            # reverse mapping: next_state -> actions (next_state_links)
            for effect in list(self.next_state_links.keys()):
                actions_for_eff = self.next_state_links.get(effect, [])
                if action in actions_for_eff:
                    actions_for_eff.remove(action)
                    if not actions_for_eff:
                        self.next_state_links.pop(effect, None)
                    else:
                        self.next_state_links[effect] = actions_for_eff

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

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem
        self.kb = FolKB(planning_problem.initial)
        self.levels = [Level(self.kb)]
        self.objects = set(arg for clause in self.kb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def __str__(self):
        levels_str = '\n'.join('Level {}:\n{}'.format(i, level)
                               for i, level in enumerate(self.levels))
        return '<Graph>\n  Objects: {}\n{}\n'.format(self.objects, levels_str)

    __repr__ = __str__

    def expand_graph(self):
        """Expands the graph by a level"""

        last_level = self.levels[-1]
        # populate state/actions/mutexes
        last_level(self.planning_problem.actions, self.objects)
        last_level.prune_invalid_actions()
        # create new level
        new_level = last_level.perform_actions()
        # populate the mutexes for the next state level to come
        new_level.state_mutexes = last_level.populate_prop_mutexes()
        self.levels.append(new_level)

    def non_mutex_goals(self, goals, index):
        """Checks whether the goals are mutually exclusive"""

        goal_perm = itertools.combinations(goals, 2)
        for g in goal_perm:
            if set(g) in self.levels[index].state_mutexes:
                return False
        return True


class GraphPlan:
    """
    Class for formulation GraphPlan algorithm
    Constructs a graph of state and action space
    Returns solution for the planning problem
    """

    def __init__(self, planning_problem):
        self.graph = Graph(planning_problem)
        self.no_goods = []
        self.solution = []

    def __str__(self):
        sol_str = ('No solution found' if not self.solution
                   else 'Solution with {} steps'.format(len(self.solution)))
        return '<GraphPlan>\n  Nogoods: {}\n  {}\n'.format(len(self.no_goods), sol_str)

    __repr__ = __str__

    def check_leveloff(self):
        """Checks if the graph has leveled off"""

        if len(self.graph.levels) < 2:
            return False

        level = self.graph.levels[-1]
        prev_level = self.graph.levels[-2]

        same_state = set(level.current_state) == set(prev_level.current_state)

        level_mutex = set(frozenset(m) for m in level.state_mutexes)
        prev_mutex = set(frozenset(m) for m in prev_level.state_mutexes)
        same_mutex = level_mutex == prev_mutex

        return same_state and same_mutex

    def get_preconditions_for(self, action_set, level):
        """Collects all unique preconditions for a given set of actions in a level"""

        all_preconditions = set()
        for action in action_set:
            preconditions = level.current_action_links.get(action, [])
            all_preconditions.update(preconditions)
        return all_preconditions

    def find_valid_action_sets(self, goals, level):
        """
        Finds sets of actions in the given level that are not mutually exclusive
        and that collectively satisfy all the goals.
        """

        valid_sets = []

        actions_for_goal = {g: level.next_state_links.get(g, []) for g in goals}
        potential_action_groups = [actions_for_goal[g] for g in goals]

        for action_combination in itertools.product(*potential_action_groups):
            action_set = set(action_combination)

            is_mutex = False
            for a1, a2 in itertools.combinations(action_set, 2):
                if {a1, a2} in level.action_mutexes:
                    is_mutex = True
                    break

            if not is_mutex and action_set not in valid_sets:
                valid_sets.append(action_set)

        return valid_sets

    def extract_solution(self, goals):
        """
        Starts the solution extraction process by calling the recursive helper
        and returning the final plan.
        """

        return self.extract_solution_recursive(set(goals), len(self.graph.levels) - 1)

    def extract_solution_recursive(self, goals, level_index):
        """
        Recursively searches for a plan backwards from a given proposition level.

        goals is the set of goal propositions to satisfy and level_index is the
        index of the proposition level currently being solved.
        """

        # Base case: we have recursed back to the initial proposition layer (level 0).
        # The goals at this point are the preconditions for the very first set of
        # actions, so we just check whether they hold in the initial state.
        if level_index == 0:
            initial_state = set(self.graph.levels[0].current_state)
            if goals.issubset(initial_state):
                return []  # success, return the empty plan to be built upon
            else:
                return None  # failure, preconditions are not met

        # Memoization: check whether we already proved this subproblem unsolvable.
        if (level_index, frozenset(goals)) in self.no_goods:
            return None

        # Recursive step: to satisfy the goals at level_index we need a set of
        # non-mutex actions from the previous level's action layer.
        action_level = self.graph.levels[level_index - 1]
        valid_action_sets = self.find_valid_action_sets(goals, action_level)

        for action_set in valid_action_sets:
            # The new sub-goals are the combined preconditions for this action set.
            new_goals = self.get_preconditions_for(action_set, action_level)

            # Recurse to solve for the new goals at the previous proposition layer.
            sub_plan = self.extract_solution_recursive(new_goals, level_index - 1)

            if sub_plan is not None:
                return sub_plan + [list(action_set)]

        # No solution from this subproblem, so record it as a no-good.
        nogood_item = (level_index, frozenset(goals))
        if nogood_item not in self.no_goods:
            self.no_goods.append(nogood_item)
        return None

    def goal_test(self, kb):
        """Return True if all of the problem's goals can be proven from the knowledge base."""
        return all(kb.ask(q) is not False for q in self.graph.planning_problem.goals)

    def execute(self):
        """Executes the GraphPlan algorithm for the given problem"""

        while True:
            self.graph.expand_graph()
            if (self.goal_test(self.graph.levels[-1].kb) and
                    self.graph.non_mutex_goals(self.graph.planning_problem.goals, -1)):
                solution = self.extract_solution(self.graph.planning_problem.goals)
                if solution:
                    return [solution]

            if self.check_leveloff():
                return None


class Linearize:
    """
    Coordinator that linearizes partially ordered solutions generated by a GraphPlan object.
    """

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem

    def filter(self, solution):
        """Filter out persistence actions from a solution"""

        new_solution = []
        for section in solution:
            new_section = []
            for operation in section:
                if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                    new_section.append(operation)
            # filter may remove all actions if all actions are persistent
            if new_section != []:
                new_solution.append(new_section)
        return new_solution

    def orderlevel(self, level, planning_problem):
        """Return valid linear order of actions for a given level"""

        for permutation in itertools.permutations(level):
            temp = copy.deepcopy(planning_problem)
            count = 0
            for action in permutation:
                try:
                    temp.act(action)
                    count += 1
                except:
                    count = 0
                    temp = copy.deepcopy(planning_problem)
                    continue
            if count == len(permutation):
                return list(permutation), temp
        # identifying a linear ordering for the level failed
        return None, planning_problem

    def execute(self):
        """Finds a total-order solution for a planning graph (not necessarily unique)"""

        graph_plan_solution = GraphPlan(self.planning_problem).execute()

        # exit if no plan found
        if graph_plan_solution is None:
            return None

        ordered_solution = None
        for possible_plan in graph_plan_solution:
            filtered_solution = self.filter(possible_plan)

            ordered_solution = []
            # planning_problem maintains the current state as we iterate over the
            # levels, allowing test application of the actions
            planning_problem = self.planning_problem
            for level in filtered_solution:
                level_solution, planning_problem = self.orderlevel(level, planning_problem)
                if not level_solution:
                    # level failed to apply, this plan does not work
                    ordered_solution = None
                    break

                for element in level_solution:
                    ordered_solution.append(element)

            if not ordered_solution:
                continue
            else:
                break

        return ordered_solution


def linearize(solution):
    """Converts a level-ordered solution into a linear solution"""

    linear_solution = []
    for section in solution[0]:
        for operation in section:
            if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                linear_solution.append(operation)

    return linear_solution


class PartialOrderPlanner:
    """
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
    8. The set of new causal links is checked for threats and if found, the threat is removed by either promotion or
       demotion. If promotion or demotion is unable to solve the problem, the planning problem cannot be solved with
       the current sequence of actions or it may not be solvable at all.
    9. These steps are repeated until the set of open preconditions is empty.
    """

    def __init__(self, planning_problem):
        self.tries = 1
        # safety bounds for the backtracking search in execute(): the maximum
        # number of actions a plan may contain (iterative-deepening target) and
        # the maximum number of node expansions per deepening level
        self._max_plan_actions = 12
        self._max_expansions = 20000
        self.planning_problem = planning_problem
        self.causal_links = []
        self.start = Action('Start', [], self.planning_problem.initial)
        self.finish = Action('Finish', self.planning_problem.goals, [])
        self.actions = set()
        self.actions.add(self.start)
        self.actions.add(self.finish)
        self.constraints = set()
        self.constraints.add((self.start, self.finish))
        self.agenda = set()
        for precond in self.finish.precond:
            self.agenda.add((precond, self.finish))
        self.expanded_actions = planning_problem.expand_actions()

    def find_open_precondition(self):
        """
        Find the open precondition with the least number of achieving actions
        (a most-constrained-variable heuristic). Returns the triple
        (precondition, action_that_needs_it, [achieving_actions]). Iteration is
        ordered deterministically so the search does not depend on set/hash
        ordering. Returns (None, None, None) when some open precondition has no
        achiever at all, which is a dead end for the current partial plan.
        """
        possible_actions = list(self.actions) + self.expanded_actions
        number_of_ways = dict()
        actions_for_precondition = dict()
        for open_precondition, act in sorted(self.agenda, key=str):
            if open_precondition in number_of_ways:
                continue
            achievers = [action for action in possible_actions
                         if any(effect == open_precondition for effect in action.effect)]
            if not achievers:
                return None, None, None
            number_of_ways[open_precondition] = len(achievers)
            actions_for_precondition[open_precondition] = achievers

        if not number_of_ways:
            return None, None, None

        chosen = min(number_of_ways, key=lambda p: (number_of_ways[p], str(p)))
        act1 = next(act for precond, act in sorted(self.agenda, key=str) if precond == chosen)
        return chosen, act1, actions_for_precondition[chosen]

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
        for action in self.planning_problem.actions:
            for effect in action.effect:
                if effect.op == oprec.op:
                    bindings = unify_mm(effect, oprec)
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
        """Generate action object given a generic action and variable bindings"""

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

        graph.update({element: set() for element in extra_elements_in_dependencies})
        while True:
            ordered = set(element for element, dependency in graph.items() if len(dependency) == 0)
            if not ordered:
                break
            yield ordered
            graph = {element: (dependency - ordered)
                     for element, dependency in graph.items()
                     if element not in ordered}
        if len(graph) != 0:
            raise ValueError('The graph is not acyclic and cannot be linearly ordered')

    def display_plan(self):
        """Display causal links, constraints and the plan"""

        print('Causal Links')
        for causal_link in self.causal_links:
            print(causal_link)

        print('\n_constraints')
        for constraint in self.constraints:
            print(constraint[0], '<', constraint[1])

        print('\n_partial Order Plan')
        print(list(reversed(list(self.toposort(self.convert(self.constraints))))))

    def execute(self, display=True):
        """
        Execute the algorithm with backtracking, using iterative deepening on the
        number of actions in the plan. The original greedy version committed to
        the first achiever it happened to iterate over and could not recover when
        that action's own preconditions turned out to be unsatisfiable, so it
        depended on hash ordering and often printed 'Probably Wrong' / "Couldn't
        find a solution". Backtracking over both action choices and threat
        resolution (promotion vs demotion), together with the deterministic
        selection in find_open_precondition and a smallest-plan-first deepening
        bound, makes the planner solve the standard problems reproducibly and
        return a short, valid plan.
        """
        pristine = self._snapshot()
        for limit in range(1, self._max_plan_actions + 1):
            self._restore(pristine)
            if self._search([self._max_expansions], limit):
                if display:
                    self.display_plan()
                else:
                    return self.constraints, self.causal_links
                return
        print("Couldn't find a solution")
        if not display:
            return None, None

    def _reachable(self, source, target):
        """True if target is forced to come after source by the ordering constraints"""

        stack, seen = [source], set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(b for a, b in self.constraints if a == node)
        return False

    def _open_threat(self):
        """
        Return an (action, causal_link) threat that is not yet resolved by the
        ordering constraints, or None if every causal link is protected. A
        causal link (a0, p, a1) is threatened by an action whose effect negates p
        unless the action is already ordered before a0 (promotion) or after a1
        (demotion).
        """
        for a0, p, a1 in self.causal_links:
            for action in self.actions:
                if action == a0 or action == a1:
                    continue
                if any(self.is_a_threat(p, effect) for effect in action.effect):
                    if not (self._reachable(action, a0) or self._reachable(a1, action)):
                        return action, (a0, p, a1)
        return None

    def _snapshot(self):
        return set(self.actions), set(self.constraints), list(self.causal_links), set(self.agenda)

    def _restore(self, snapshot):
        self.actions, self.constraints, self.causal_links, self.agenda = (
            set(snapshot[0]), set(snapshot[1]), list(snapshot[2]), set(snapshot[3]))

    def _search(self, budget, limit):
        """
        Recursively complete the partial plan, backtracking on failure. Three
        kinds of choice points are explored: which action satisfies an open
        precondition, how each threat is resolved (promotion vs demotion), and -
        bounded by 'limit' - whether to introduce a new action at all. Returns
        True and leaves the solution in self.* on success.
        """
        if budget[0] <= 0:
            return False
        budget[0] -= 1

        # first, resolve any outstanding threat to a causal link (choice point)
        threat = self._open_threat()
        if threat is not None:
            action, (a0, p, a1) = threat
            snapshot = self._snapshot()
            for ordering in ((action, a0), (a1, action)):  # promotion, then demotion
                new_constraints = self.add_const(ordering, self.constraints)
                if ordering in new_constraints:  # ordering was consistent (acyclic and allowed)
                    self.constraints = new_constraints
                    if self._search(budget, limit):
                        return True
                self._restore(snapshot)
            return False

        # no open threats: a plan with an empty agenda is a complete solution
        if not self.agenda:
            return True

        # select <G, act1> from the agenda (most-constrained precondition first)
        G, act1, possible_actions = self.find_open_precondition()
        if G is None:  # an open precondition has no achiever -> dead end
            return False

        # number of actions already introduced, excluding the dummy Start/Finish
        introduced = len(self.actions) - 2
        snapshot = self._snapshot()
        # try each achiever deterministically, reusing existing actions first
        for act0 in sorted(set(possible_actions), key=lambda a: (a not in self.actions, str(a))):
            is_new = act0 not in self.actions
            if is_new and introduced >= limit:  # deepening bound on plan size
                continue
            self.agenda.discard((G, act1))
            self.actions.add(act0)
            self.constraints = self.add_const((self.start, act0), self.constraints)
            self.constraints = self.add_const((act0, act1), self.constraints)
            # the causal link act0 --G--> act1 requires act0 strictly before act1
            # (and after start); add_const drops an ordering that would create a
            # cycle, so reject the choice when the required ordering is not enforced
            if ((act0 == act1 or self._reachable(act0, act1)) and
                    (act0 == self.start or self._reachable(self.start, act0))):
                if (act0, G, act1) not in self.causal_links:
                    self.causal_links.append((act0, G, act1))
                if is_new:  # a freshly introduced action contributes its own preconditions
                    for precondition in act0.precond:
                        self.agenda.add((precondition, act0))
                if self._search(budget, limit):
                    return True
            # undo and try the next achiever
            self._restore(snapshot)
        return False


def spare_tire_graph_plan():
    """Solves the spare tire problem using GraphPlan"""
    return GraphPlan(spare_tire()).execute()


def three_block_tower_graph_plan():
    """Solves the Sussman Anomaly problem using GraphPlan"""
    return GraphPlan(three_block_tower()).execute()


def air_cargo_graph_plan():
    """Solves the air cargo problem using GraphPlan"""
    return GraphPlan(air_cargo()).execute()


def have_cake_and_eat_cake_too_graph_plan():
    """Solves the cake problem using GraphPlan"""
    return GraphPlan(have_cake_and_eat_cake_too()).execute()


def shopping_graph_plan():
    """Solves the shopping problem using GraphPlan"""
    return GraphPlan(shopping_problem()).execute()


def socks_and_shoes_graph_plan():
    """Solves the socks and shoes problem using GraphPlan"""
    return GraphPlan(socks_and_shoes()).execute()


def simple_blocks_world_graph_plan():
    """Solves the simple blocks world problem"""
    return GraphPlan(simple_blocks_world()).execute()


class HLA(Action):
    """
    Define Actions for the real-world (that may be refined further), and satisfy resource
    constraints.
    """
    unique_group = 1

    def __init__(self, action, precond=None, effect=None, duration=0, consume=None, use=None):
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


class RealWorldPlanningProblem(PlanningProblem):
    """
    Define real-world problems by aggregating resources as numerical quantities instead of
    named entities.

    This class is identical to PDDL, except that it overloads the act function to handle
    resource and ordering conditions imposed by HLA as opposed to Action.
    """

    def __init__(self, initial, goals, actions, jobs=None, resources=None):
        super().__init__(initial, goals, actions)
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
        self.initial = list_action.do_action(self.jobs, self.resources, self.initial, args).clauses

    def refinements(self, library):  # refinements may be (multiple) HLA themselves ...
        """
        State is a Problem, containing the current state kb library is a
        dictionary containing details for every possible refinement. e.g.::

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
                ]}
        """
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == self.name]
        for i in indices:
            actions = []
            for j in range(len(library['steps'][i])):
                # find the index of the step [j]  of the HLA
                index_step = [k for k, x in enumerate(library['HLA']) if x == library['steps'][i][j]][0]
                precond = library['precond'][index_step][0]  # preconditions of step [j]
                effect = library['effect'][index_step][0]  # effect of step [j]
                actions.append(HLA(library['steps'][i][j], precond, effect))
            yield actions

    def hierarchical_search(self, hierarchy, max_depth=None):
        """
        [Figure 11.5]
        'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world problem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details).
        With the default max_depth=None the search follows Figure 11.5 exactly and, like the
        textbook algorithm, may not terminate on a recursive (cyclic) hierarchy whose goal is
        unreachable. Passing max_depth prunes any plan longer than that many actions, which
        guarantees termination (returning None when no solution is found within the bound).
        """
        act = Node(self.initial, None, [self.actions[0]])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            # finds the first non primitive hla in plan actions
            (hla, index) = RealWorldPlanningProblem.find_hla(plan, hierarchy)
            prefix = plan.action[:index]
            outcome = RealWorldPlanningProblem(
                RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
            suffix = plan.action[index + 1:]
            if not hla:  # hla is None and plan is primitive
                if outcome.goal_test():
                    return plan.action
            else:
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):  # find refinements
                    refined_plan = prefix + sequence + suffix
                    if max_depth is None or len(refined_plan) <= max_depth:
                        frontier.append(Node(outcome.initial, plan, refined_plan))

    def result(state, actions):
        """The outcome of applying an action to the current problem"""
        for a in actions:
            if a.check_precond(state, a.args):
                state = a(state, a.args).clauses
        return state

    def angelic_search(self, hierarchy, initial_plan):
        """
        [Figure 11.8]
        A hierarchical planning algorithm that uses angelic semantics to identify and
        commit to high-level plans that work while avoiding high-level plans that don’t.
        The predicate MAKING-PROGRESS checks to make sure that we aren’t stuck in an infinite regression
        of refinements.
        At top level, call ANGELIC-SEARCH with [Act] as the initial_plan.

        InitialPlan contains a sequence of HLA's with angelic semantics

        The possible effects of an angelic HLA in initial_plan are:
        ~ : effect remove
        $+: effect possibly add
        $-: effect possibly remove
        $$: possibly add or remove
        """
        frontier = deque(initial_plan)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()  # sequence of HLA/Angelic HLA's
            opt_reachable_set = RealWorldPlanningProblem.reach_opt(self.initial, plan)
            pes_reachable_set = RealWorldPlanningProblem.reach_pes(self.initial, plan)
            if self.intersects_goal(opt_reachable_set):
                if RealWorldPlanningProblem.is_primitive(plan, hierarchy):
                    return [x for x in plan.action]
                guaranteed = self.intersects_goal(pes_reachable_set)
                if guaranteed and RealWorldPlanningProblem.making_progress(plan, initial_plan):
                    final_state = guaranteed[0]  # any element of guaranteed
                    return RealWorldPlanningProblem.decompose(hierarchy, final_state, pes_reachable_set)
                # there should be at least one HLA/AngelicHLA, otherwise plan would be primitive
                hla, index = RealWorldPlanningProblem.find_hla(plan, hierarchy)
                prefix = plan.action[:index]
                suffix = plan.action[index + 1:]
                outcome = RealWorldPlanningProblem(
                    RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):  # find refinements
                    frontier.append(
                        AngelicNode(outcome.initial, plan, prefix + sequence + suffix, prefix + sequence + suffix))

    def intersects_goal(self, reachable_set):
        """
        Find the intersection of the reachable states and the goal
        """
        return [y for x in list(reachable_set.keys())
                for y in reachable_set[x]
                if all(goal in y for goal in self.goals)]

    def is_primitive(plan, library):
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
        optimistic_description = plan.action  # list of angelic actions with optimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, optimistic_description)

    def reach_pes(init, plan):
        """
        Finds the pessimistic reachable set of the sequence of actions in plan
        """
        reachable_set = {0: [init]}
        pessimistic_description = plan.action_pes  # list of angelic actions with pessimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, pessimistic_description)

    def find_reachable_set(reachable_set, action_description):
        """
        Finds the reachable states of the action_description when applied in each state of reachable set.
        """
        for i in range(len(action_description)):
            reachable_set[i + 1] = []
            if type(action_description[i]) is AngelicHLA:
                possible_actions = action_description[i].angelic_action()
            else:
                possible_actions = action_description
            for action in possible_actions:
                for state in reachable_set[i]:
                    if action.check_precond(state, action.args):
                        if action.effect[0]:
                            new_state = action(state, action.args).clauses
                            reachable_set[i + 1].append(new_state)
                        else:
                            reachable_set[i + 1].append(state)
        return reachable_set

    def find_hla(plan, hierarchy):
        """
        Finds the the first HLA action in plan.action, which is not primitive
        and its corresponding index in plan.action
        """
        hla = None
        index = len(plan.action)
        for i in range(len(plan.action)):  # find the first HLA in plan, that is not primitive
            if not RealWorldPlanningProblem.is_primitive(Node(plan.state, plan.parent, [plan.action[i]]), hierarchy):
                hla = plan.action[i]
                index = i
                break
        return hla, index

    def making_progress(plan, initial_plan):
        """
        Prevents from infinite regression of refinements

        (infinite regression of refinements happens when the algorithm finds a plan that
        its pessimistic reachable set intersects the goal inside a call to decompose on
        the same plan, in the same circumstances)
        """
        for i in range(len(initial_plan)):
            if plan == initial_plan[i]:
                return False
        return True

    def decompose(hierarchy, plan, s_f, reachable_set):
        """
        Recursively refine the high-level actions of an abstract plan into a concrete
        sequence of primitive actions. Working backwards from the final state ``s_f``,
        it picks an intermediate state for each pessimistic action from ``reachable_set``
        and uses angelic search to expand it, returning the assembled primitive solution
        (or None if some action cannot be refined).
        """
        solution = []
        i = max(reachable_set.keys())
        while plan.action_pes:
            action = plan.action_pes.pop()
            if i == 0:
                return solution
            s_i = RealWorldPlanningProblem.find_previous_state(s_f, reachable_set, i, action)
            problem = RealWorldPlanningProblem(s_i, s_f, plan.action)
            angelic_call = RealWorldPlanningProblem.angelic_search(problem, hierarchy,
                                                                   [AngelicNode(s_i, Node(None), [action], [action])])
            if angelic_call:
                for x in angelic_call:
                    solution.insert(0, x)
            else:
                return None
            s_f = s_i
            i -= 1
        return solution

    def find_previous_state(s_f, reachable_set, i, action):
        """
        Given a final state s_f and an action finds a state s_i in reachable_set
        such that when action is applied to state s_i returns s_f.
        """
        s_i = reachable_set[i - 1][0]
        for state in reachable_set[i - 1]:
            if s_f in [x for x in RealWorldPlanningProblem.reach_pes(
                    state, AngelicNode(state, None, [action], [action]))[1]]:
                s_i = state
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
    add_wheels1 = HLA('AddWheels1', precond='~Has(C1, W1)', effect='Has(C1, W1)', duration=30, use={'WheelStations': 1},
                      consume={'LugNuts': 20})
    add_wheels2 = HLA('AddWheels2', precond='~Has(C2, W2)', effect='Has(C2, W2)', duration=15, use={'WheelStations': 1},
                      consume={'LugNuts': 20})
    inspect1 = HLA('Inspect1', precond='~Inspected(C1)', effect='Inspected(C1)', duration=10, use={'Inspectors': 1})
    inspect2 = HLA('Inspect2', precond='~Inspected(C2)', effect='Inspected(C2)', duration=10, use={'Inspectors': 1})

    actions = [add_engine1, add_engine2, add_wheels1, add_wheels2, inspect1, inspect2]

    job_group1 = [add_engine1, add_wheels1, inspect1]
    job_group2 = [add_engine2, add_wheels2, inspect2]

    return RealWorldPlanningProblem(
        initial='Car(C1) & Car(C2) & Wheels(W1) & Wheels(W2) & Engine(E2) & Engine(E2) & ~Has(C1, E1) & ~Has(C2, '
                'E2) & ~Has(C1, W1) & ~Has(C2, W2) & ~Inspected(C1) & ~Inspected(C2)',
        goals='Has(C1, W1) & Has(C1, E1) & Inspected(C1) & Has(C2, W2) & Has(C2, E2) & Inspected(C2)',
        actions=actions,
        jobs=[job_group1, job_group2],
        resources=resources)


def go_to_sfo():
    """Go to SFO Problem"""

    go_home_sfo1 = HLA('Go(Home, SFO)', precond='At(Home) & Have(Car)', effect='At(SFO) & ~At(Home)')
    go_home_sfo2 = HLA('Go(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
    drive_home_sfoltp = HLA('Drive(Home, SFOLongTermParking)', precond='At(Home) & Have(Car)',
                            effect='At(SFOLongTermParking) & ~At(Home)')
    shuttle_sfoltp_sfo = HLA('Shuttle(SFOLongTermParking, SFO)', precond='At(SFOLongTermParking)',
                             effect='At(SFO) & ~At(SFOLongTermParking)')
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
            ['At(SFO) & ~At(Home)']]}

    return RealWorldPlanningProblem(initial='At(Home)', goals='At(SFO)', actions=actions), library


class AngelicHLA(HLA):
    """
    Define Actions for the real-world (that may be refined further), under angelic semantics
    """

    def __init__(self, action, precond, effect, duration=0, consume=None, use=None):
        super().__init__(action, precond, effect, duration, consume, use)

    def convert(self, clauses):
        """
        Converts strings into Exprs
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable)
        and furthermore can have following effects on the variables::

            Possibly add variable    ( $+ )
            Possibly remove variable ( $- )
            Possibly add or remove a variable ( $$ )

        Overrides HLA.convert function
        """
        lib = {'~': 'Not',
               '$+': 'PosYes',
               '$-': 'PosNot',
               '$$': 'PosYesNot'}

        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                for ch in lib.keys():
                    if clauses[i].op == ch:
                        clauses[i] = expr(lib[ch] + str(clauses[i].args[0]))

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


            example: the angelic action with effects possibly add A and possibly add or remove B corresponds to the
            following 6 effects of HLAs:


            '$+A & $$B':    HLA_1: 'A & B'   (add A and add B)
                            HLA_2: 'A & ~B'  (add A and remove B)
                            HLA_3: 'A'       (add A)
                            HLA_4: 'B'       (add B)
                            HLA_5: '~B'      (remove B)
                            HLA_6: ' '       (no effect)

        """

        effects = [[]]
        for clause in self.effect:
            (n, w) = AngelicHLA.compute_parameters(clause)
            effects = effects * n  # create n copies of effects
            it = range(1)
            if len(effects) != 0:
                # split effects into n sublists (separate n copies created in compute_parameters)
                it = range(len(effects) // n)
            for i in it:
                if effects[i]:
                    if clause.args:
                        effects[i] = expr(str(effects[i]) + '&' + str(
                            Expr(clause.op[w:], clause.args[0])))  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(Expr(clause.op[6:], clause.args[0])))
                    else:
                        effects[i] = expr(
                            str(effects[i]) + '&' + str(expr(clause.op[w:])))  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(expr(clause.op[6:])))

                else:
                    if clause.args:
                        effects[i] = Expr(clause.op[w:], clause.args[0])  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = Expr(clause.op[6:], clause.args[0])

                    else:
                        effects[i] = expr(clause.op[w:])  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(clause.op[6:])

        return [HLA(Expr(self.name, self.args), self.precond, effects[i]) for i in range(len(effects))]

    def compute_parameters(clause):
        """
        computes n,w

        n = number of HLA effects that the angelic HLA corresponds to
        w = length of representation of angelic HLA effect

                    n = 1, if effect is add
                    n = 1, if effect is remove
                    n = 2, if effect is possibly add
                    n = 2, if effect is possibly remove
                    n = 3, if effect is possibly add or remove

        """
        if clause.op[:9] == 'PosYesNot':
            # possibly add/remove variable: three possible effects for the variable
            n = 3
            w = 9
        elif clause.op[:6] == 'PosYes':  # possibly add variable: two possible effects for the variable
            n = 2
            w = 6
        elif clause.op[:6] == 'PosNot':  # possibly remove variable: two possible effects for the variable
            n = 2
            w = 3  # We want to keep 'Not' from 'PosNot' when adding action
        else:  # variable or ~variable
            n = 1
            w = 0
        return n, w


class AngelicNode(Node):
    """
    Extends the class Node.
    self.action:     contains the optimistic description of an angelic HLA
    self.action_pes: contains the pessimistic description of an angelic HLA
    """

    def __init__(self, state, parent=None, action_opt=None, action_pes=None, path_cost=0):
        super().__init__(state, parent, action_opt, path_cost)
        self.action_pes = action_pes
