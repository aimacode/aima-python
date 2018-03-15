"""Planning (Chapters 10-11)
"""

import copy
import itertools
from search import Node
from utils import Expr, expr, first, FIFOQueue, partition
from logic import FolKB
from logic import fol_bc_and


class PlanningKB:
    """ A PlanningKB contains a set of Expr objects that are immutable and hashable.
     With its goal clauses and its accompanying h function, the KB
     can be used by the A* algorithm in its search Nodes. (search.py) """
    def __init__(self, goals, initial_clauses=None):
        if initial_clauses is None:
            initial_clauses = []
        self.goal_clauses = frozenset(goals)
        self.clause_set = frozenset(initial_clauses)

    def __eq__(self, other):
        """search.Node has a __eq__ method for each state, so this method must be implemented too."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.clause_set == other.clause_set

    def __lt__(self, other):
        """Goals must be part of each PlanningKB because search.Node has a __lt__ method that compares state to state
        (used for ordering the priority queue). As a result, states must be compared by how close they are to the goal
        using a heuristic."""
        if not isinstance(other, self.__class__):
            return NotImplementedError

        # heuristic is whether there are fewer unresolved goals in the current KB than the other KB.
        return len(self.goal_clauses - self.clause_set) < len(self.goal_clauses - other.clause_set)

    def __hash__(self):
        """search.Node has a __hash__ method for each state, so this method must be implemented too."""
        return hash(self.clause_set)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, list(self.goal_clauses), list(self.clause_set))

    def goal_test(self):
        """ Goal is satisfied when KB at least contains all goal clauses. """
        return self.clause_set >= self.goal_clauses

    def h(self):
        """ Returns: number of remaining goal clauses to be satisfied """
        return len(self.goal_clauses - self.clause_set)

    def fetch_rules_for_goal(self, goal):
        return self.clause_set

class PDDL:
    """
    Planning Domain Definition Language (PDDL) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial_state, actions, goal_test):
        self.kb = FolKB(initial_state)
        self.actions = actions
        self.goal_test_func = goal_test

    def goal_test(self):
        return self.goal_test_func(self.kb)

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
        if not list_action.check_precond(self.kb, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        list_action(self.kb, args)


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PDDL.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negated literals.
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("~Eaten(food)")]
    effect = [expr("Eaten(food)"), expr("~Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, expression, precond, effect):
        self.name = expression.op
        self.args = expression.args
        self.subst = None

        def is_negative_clause(e):
            return e.op == '~' and len(e.args) == 1

        precond_neg, precond_pos = partition(precond, is_negative_clause)
        self.precond_pos = set(precond_pos)
        self.precond_neg = set(e.args[0] for e in precond_neg)  # change the negative Exprs to positive
        effect_rem, effect_add = partition(effect, is_negative_clause)
        self.effect_add = set(effect_add)
        self.effect_rem = set(e.args[0] for e in effect_rem)  # change the negative Exprs to positive

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, Expr(self.name, self.args),
                                       list(self.precond_pos) + ['~{0}'.format(p) for p in self.precond_neg],
                                       list(self.effect_add) + ['~{0}'.format(e) for e in self.effect_rem])

    def substitute(self, subst, e):
        """Replaces variables in expression with the same substitution used for the precondition. """
        new_args = [subst.get(x, x) for x in e.args]
        return Expr(e.op, *new_args)

    def check_neg_precond(self, kb, precond, subst):
        for s in subst:
            for _ in fol_bc_and(kb, list(precond), s):
                # if any negative preconditions are satisfied by the substitution, then exit loop.
                if precond:
                    break
            else:
                neg_precond = frozenset(self.substitute(s, x) for x in precond)
                clause_set = kb.fetch_rules_for_goal(None)
                # negative preconditions succeed if none of them are found in the KB.
                if clause_set.isdisjoint(neg_precond):
                    yield s

    def check_pos_precond(self, kb, precond, subst):
        clause_set = kb.fetch_rules_for_goal(None)
        for s in fol_bc_and(kb, list(precond), subst):
            pos_precond = frozenset(self.substitute(s, x) for x in precond)
            # are all preconds found in the KB?
            if clause_set.issuperset(pos_precond):
                yield s

    def check_precond(self, kb):
        """Checks if preconditions are satisfied in the current state"""
        yield from self.check_neg_precond(kb, self.precond_neg, self.check_pos_precond(kb, self.precond_pos, {}))

    def act(self, subst, kb):
        new_kb = PlanningKB(kb.goal_clauses, kb.clause_set)
        """Executes the action on the state's kb"""
        clause_set = set(new_kb.clause_set)
        # remove negative literals
        for clause in self.effect_rem:
            subst_clause = self.substitute(subst, clause)
            clause_set.discard(subst_clause)
        # add positive literals
        for clause in self.effect_add:
            subst_clause = self.substitute(subst, clause)
            clause_set.add(subst_clause)
        new_kb.clause_set = frozenset(clause_set)
        return new_kb


def air_cargo():
    goals = [expr('At(C1, JFK)'), expr('At(C2, SFO)')]

    init = PlanningKB(goals,
                      [expr('At(C1, SFO)'),
                       expr('At(C2, JFK)'),
                       expr('At(P1, SFO)'),
                       expr('At(P2, JFK)'),
                       expr('Cargo(C1)'),
                       expr('Cargo(C2)'),
                       expr('Plane(P1)'),
                       expr('Plane(P2)'),
                       expr('Airport(JFK)'),
                       expr('Airport(SFO)')])

    # Actions

    #  Load
    precond = [expr('At(c, a)'), expr('At(p, a)'), expr('Cargo(c)'), expr('Plane(p)'), expr('Airport(a)')]
    effect = [expr('In(c, p)'), expr('~At(c, a)')]
    load = Action(expr('Load(c, p, a)'), precond, effect)

    #  Unload
    precond = [expr('In(c, p)'), expr('At(p, a)'), expr('Cargo(c)'), expr('Plane(p)'), expr('Airport(a)')]
    effect = [expr('At(c, a)'), expr('~In(c, p)')]
    unload = Action(expr('Unload(c, p, a)'), precond, effect)

    #  Fly
    #  Used used 'f' instead of 'from' because 'from' is a python keyword and expr uses eval() function
    precond = [expr('At(p, f)'), expr('Plane(p)'), expr('Airport(f)'), expr('Airport(to)')]
    effect = [expr('At(p, to)'), expr('~At(p, f)')]
    fly = Action(expr('Fly(p, f, to)'), precond, effect)

    return PDDL(init, [load, unload, fly], goals)


def spare_tire():
    goals = [expr('At(Spare, Axle)')]

    init = PlanningKB(goals,
                      [expr('At(Flat, Axle)'),
                       expr('At(Spare, Trunk)')])

    # Actions
    #  Remove(Spare, Trunk)
    precond = [expr('At(Spare, Trunk)')]
    effect = [expr('At(Spare, Ground)'), expr('~At(Spare, Trunk)')]
    remove_spare = Action(expr('Remove(Spare, Trunk)'), precond, effect)

    #  Remove(Flat, Axle)
    precond = [expr('At(Flat, Axle)')]
    effect = [expr('At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    remove_flat = Action(expr('Remove(Flat, Axle)'), precond, effect)

    #  PutOn(Spare, Axle)
    precond = [expr('At(Spare, Ground)'), expr('~At(Flat, Axle)')]
    effect = [expr('At(Spare, Axle)'), expr('~At(Spare, Ground)')]
    put_on_spare = Action(expr('PutOn(Spare, Axle)'), precond, effect)

    #  LeaveOvernight
    precond = []
    effect = [expr('~At(Spare, Ground)'), expr('~At(Spare, Axle)'), expr('~At(Spare, Trunk)'),
              expr('~At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    leave_overnight = Action(expr('LeaveOvernight'), precond, effect)

    return PDDL(init, [remove_spare, remove_flat, put_on_spare, leave_overnight], goals)

def three_block_tower():
    goals = [expr('On(A, B)'), expr('On(B, C)')]
    init = PlanningKB(goals,
                      [expr('On(A, Table)'),
                       expr('On(B, Table)'),
                       expr('On(C, Table)'),
                       expr('Block(A)'),
                       expr('Block(B)'),
                       expr('Block(C)'),
                       expr('Clear(A)'),
                       expr('Clear(B)'),
                       expr('Clear(C)')])

    # Actions
    #  Move(b, x, y)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Clear(y)'), expr('Block(b)')]
    effect = [expr('On(b, y)'), expr('Clear(x)'), expr('~On(b, x)'), expr('~Clear(y)')]
    move = Action(expr('Move(b, x, y)'), precond, effect)

    #  MoveToTable(b, x)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    effect = [expr('On(b, Table)'), expr('Clear(x)'), expr('~On(b, x)')]
    move_to_table = Action(expr('MoveToTable(b, x)'), precond, effect)

    return PDDL(init, [move, move_to_table], goals)


def have_cake_and_eat_cake_too():
    init = [expr('Have(Cake)')]

    def goal_test(kb):
        required = [expr('Have(Cake)'), expr('Eaten(Cake)')]
        return all(kb.ask(q) is not False for q in required)

    # Actions

    # Eat cake
    precond = [expr('Have(Cake)')]
    effect = [expr('Eaten(Cake)'), expr('Have(Cake)')]
    eat_cake = Action(expr('Eat(Cake)'), precond, effect)

    # Bake Cake
    precond = [expr('Have(Cake)')]
    effect = [expr('Have(Cake)')]
    bake_cake = Action(expr('Bake(Cake)'), precond, effect)

    return PDDL(init, [eat_cake, bake_cake], goal_test)

def sussman_anomaly():
    goals = [expr('On(A, B)'), expr('On(B, C)')]
    init = PlanningKB(goals,
                      [expr('On(A, Table)'),
                       expr('On(B, Table)'),
                       expr('On(C, A)'),
                       expr('Block(A)'),
                       expr('Block(B)'),
                       expr('Block(C)'),
                       expr('Clear(B)'),
                       expr('Clear(C)')])

    # Actions
    #  Move(b, x, y)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Clear(y)'), expr('Block(b)')]
    effect = [expr('On(b, y)'), expr('Clear(x)'), expr('~On(b, x)'), expr('~Clear(y)')]
    move = Action(expr('Move(b, x, y)'), precond, effect)

    #  MoveToTable(b, x)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    effect = [expr('On(b, Table)'), expr('Clear(x)'), expr('~On(b, x)')]
    move_to_table = Action(expr('MoveToTable(b, x)'), precond, effect)

    return PDDL(init, [move, move_to_table], goals)


class Level():
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, poskb, negkb):
        self.poskb = poskb
        # Current state
        self.current_state_pos = poskb.clauses
        self.current_state_neg = negkb.clauses
        # Current action to current state link
        self.current_action_links_pos = {}
        self.current_action_links_neg = {}
        # Current state to action link
        self.current_state_links_pos = {}
        self.current_state_links_neg = {}
        # Current action to next state link
        self.next_action_links = {}
        # Next state to current action link
        self.next_state_links_pos = {}
        self.next_state_links_neg = {}
        self.mutex = []

    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()

    def find_mutex(self):
        # Inconsistent effects
        for poseff in self.next_state_links_pos:
            negeff = poseff
            if negeff in self.next_state_links_neg:
                for a in self.next_state_links_pos[poseff]:
                    for b in self.next_state_links_neg[negeff]:
                        if {a, b} not in self.mutex:
                            self.mutex.append({a, b})

        # Interference
        for posprecond in self.current_state_links_pos:
            negeff = posprecond
            if negeff in self.next_state_links_neg:
                for a in self.current_state_links_pos[posprecond]:
                    for b in self.next_state_links_neg[negeff]:
                        if {a, b} not in self.mutex:
                            self.mutex.append({a, b})

        for negprecond in self.current_state_links_neg:
            poseff = negprecond
            if poseff in self.next_state_links_pos:
                for a in self.next_state_links_pos[poseff]:
                    for b in self.current_state_links_neg[negprecond]:
                        if {a, b} not in self.mutex:
                            self.mutex.append({a, b})

        # Competing needs
        for posprecond in self.current_state_links_pos:
            negprecond = posprecond
            if negprecond in self.current_state_links_neg:
                for a in self.current_state_links_pos[posprecond]:
                    for b in self.current_state_links_neg[negprecond]:
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

        self.mutex = self.mutex+state_mutex

    def build(self, actions, objects):

        # Add persistence actions for positive states
        for clause in self.current_state_pos:
            self.current_action_links_pos[Expr('Persistence', clause)] = [clause]
            self.next_action_links[Expr('Persistence', clause)] = [clause]
            self.current_state_links_pos[clause] = [Expr('Persistence', clause)]
            self.next_state_links_pos[clause] = [Expr('Persistence', clause)]

        # Add persistence actions for negative states
        for clause in self.current_state_neg:
            not_expr = Expr('not'+clause.op, clause.args)
            self.current_action_links_neg[Expr('Persistence', not_expr)] = [clause]
            self.next_action_links[Expr('Persistence', not_expr)] = [clause]
            self.current_state_links_neg[clause] = [Expr('Persistence', not_expr)]
            self.next_state_links_neg[clause] = [Expr('Persistence', not_expr)]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.poskb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower():
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links_pos[new_action] = []
                    self.current_action_links_neg[new_action] = []

                    for clause in a.precond_pos:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links_pos[new_action].append(new_clause)
                        if new_clause in self.current_state_links_pos:
                            self.current_state_links_pos[new_clause].append(new_action)
                        else:
                            self.current_state_links_pos[new_clause] = [new_action]

                    for clause in a.precond_neg:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links_neg[new_action].append(new_clause)
                        if new_clause in self.current_state_links_neg:
                            self.current_state_links_neg[new_clause].append(new_action)
                        else:
                            self.current_state_links_neg[new_clause] = [new_action]

                    self.next_action_links[new_action] = []
                    for clause in a.effect_add:
                        new_clause = a.substitute(clause, arg)
                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links_pos:
                            self.next_state_links_pos[new_clause].append(new_action)
                        else:
                            self.next_state_links_pos[new_clause] = [new_action]

                    for clause in a.effect_rem:
                        new_clause = a.substitute(clause, arg)
                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links_neg:
                            self.next_state_links_neg[new_clause].append(new_action)
                        else:
                            self.next_state_links_neg[new_clause] = [new_action]

    def perform_actions(self):
        new_kb_pos = FolKB(list(set(self.next_state_links_pos.keys())))
        new_kb_neg = FolKB(list(set(self.next_state_links_neg.keys())))

        return Level(new_kb_pos, new_kb_neg)


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, pddl, negkb):
        self.pddl = pddl
        self.levels = [Level(pddl.kb, negkb)]
        self.objects = set(arg for clause in pddl.kb.clauses + negkb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def expand_graph(self):
        last_level = self.levels[-1]
        last_level(self.pddl.actions, self.objects)
        self.levels.append(last_level.perform_actions())

    def non_mutex_goals(self, goals, index):
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

    def __init__(self, pddl, negkb):
        self.graph = Graph(pddl, negkb)
        self.nogoods = []
        self.solution = []

    def check_leveloff(self):
        first_check = (set(self.graph.levels[-1].current_state_pos) ==
                       set(self.graph.levels[-2].current_state_pos))
        second_check = (set(self.graph.levels[-1].current_state_neg) ==
                        set(self.graph.levels[-2].current_state_neg))

        if first_check and second_check:
            return True

    def extract_solution(self, goals_pos, goals_neg, index):
        level = self.graph.levels[index]
        if not self.graph.non_mutex_goals(goals_pos+goals_neg, index):
            self.nogoods.append((level, goals_pos, goals_neg))
            return

        level = self.graph.levels[index-1]

        # Create all combinations of actions that satisfy the goal
        actions = []
        for goal in goals_pos:
            actions.append(level.next_state_links_pos[goal])

        for goal in goals_neg:
            actions.append(level.next_state_links_neg[goal])

        all_actions = list(itertools.product(*actions))

        # Filter out the action combinations which contain mutexes
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

                new_goals_pos = []
                new_goals_neg = []
                for act in set(action_list):
                    if act in level.current_action_links_pos:
                        new_goals_pos = new_goals_pos + level.current_action_links_pos[act]

                for act in set(action_list):
                    if act in level.current_action_links_neg:
                        new_goals_neg = new_goals_neg + level.current_action_links_neg[act]

                if abs(index)+1 == len(self.graph.levels):
                    return
                elif (level, new_goals_pos, new_goals_neg) in self.nogoods:
                    return
                else:
                    self.extract_solution(new_goals_pos, new_goals_neg, index-1)

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


def spare_tire_graphplan():
    pddl = spare_tire()
    negkb = FolKB([expr('At(Flat, Trunk)')])
    graphplan = GraphPlan(pddl, negkb)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    # Not sure
    goals_pos = [expr('At(Spare, Axle)'), expr('At(Flat, Ground)')]
    goals_neg = []

    while True:
        if (goal_test(graphplan.graph.levels[-1].poskb, goals_pos) and
                graphplan.graph.non_mutex_goals(goals_pos+goals_neg, -1)):
            solution = graphplan.extract_solution(goals_pos, goals_neg, -1)
            if solution:
                return solution
        graphplan.graph.expand_graph()
        if len(graphplan.graph.levels) >=2 and graphplan.check_leveloff():
            return None


def double_tennis_problem():
    init = [expr('At(A, LeftBaseLine)'),
            expr('At(B, RightNet)'),
            expr('Approaching(Ball, RightBaseLine)'),
            expr('Partner(A, B)'),
            expr('Partner(B, A)')]

    def goal_test(kb):
        required = [expr('Goal(Returned(Ball))'), expr('At(a, RightNet)'), expr('At(a, LeftNet)')]
        return all(kb.ask(q) is not False for q in required)

    # Actions

    # Hit
    precond_pos = [expr("Approaching(Ball,loc)"), expr("At(actor,loc)")]
    precond_neg = []
    effect_add = [expr("Returned(Ball)")]
    effect_rem = []
    hit = Action(expr("Hit(actor, Ball)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    # Go
    precond_pos = [expr("At(actor, loc)")]
    precond_neg = []
    effect_add = [expr("At(actor, to)")]
    effect_rem = [expr("At(actor, loc)")]
    go = Action(expr("Go(actor, to)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDDL(init, [hit, go], goal_test)


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
        precond = precond or [None, None]
        effect = effect or [None, None]
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
        super().act(kb, args)  # update knowledge base
        for resource in self.consumes:  # remove consumed resources
            available_resources[resource] -= self.consumes[resource]
        self.completed = True  # set the task status to complete

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


class Problem(PDDL):
    """
    Define real-world problems by aggregating resources as numerical quantities instead of
    named entities.

    This class is identical to PDLL, except that it overloads the act function to handle
    resource and ordering conditions imposed by HLA as opposed to Action.
    """
    def __init__(self, initial_state, actions, goal_test, jobs=None, resources=None):
        super().__init__(initial_state, actions, goal_test)
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
        list_action.do_action(self.jobs, self.resources, self.kb, args)

    def refinements(hla, state, library):  # TODO - refinements may be (multiple) HLA themselves ...
        """
        state is a Problem, containing the current state kb
        library is a dictionary containing details for every possible refinement. eg:
        {
        "HLA": [
            "Go(Home,SFO)",
            "Go(Home,SFO)",
            "Drive(Home, SFOLongTermParking)",
            "Shuttle(SFOLongTermParking, SFO)",
            "Taxi(Home, SFO)"
               ],
        "steps": [
            ["Drive(Home, SFOLongTermParking)", "Shuttle(SFOLongTermParking, SFO)"],
            ["Taxi(Home, SFO)"],
            [], # empty refinements ie primitive action
            [],
            []
               ],
        "precond_pos": [
            ["At(Home), Have(Car)"],
            ["At(Home)"],
            ["At(Home)", "Have(Car)"]
            ["At(SFOLongTermParking)"]
            ["At(Home)"]
                       ],
        "precond_neg": [[],[],[],[],[]],
        "effect_pos": [
            ["At(SFO)"],
            ["At(SFO)"],
            ["At(SFOLongTermParking)"],
            ["At(SFO)"],
            ["At(SFO)"]
                      ],
        "effect_neg": [
            ["At(Home)"],
            ["At(Home)"],
            ["At(Home)"],
            ["At(SFOLongTermParking)"],
            ["At(Home)"]
                      ]
        }
        """
        e = Expr(hla.name, hla.args)
        indices = [i for i, x in enumerate(library["HLA"]) if expr(x).op == hla.name]
        for i in indices:
            action = HLA(expr(library["steps"][i][0]), [  # TODO multiple refinements
                    [expr(x) for x in library["precond_pos"][i]],
                    [expr(x) for x in library["precond_neg"][i]]
                ],
                [
                    [expr(x) for x in library["effect_pos"][i]],
                    [expr(x) for x in library["effect_neg"][i]]
                ])
            if action.check_precond(state.kb, action.args):
                yield action

    def hierarchical_search(problem, hierarchy):
        """
        [Figure 11.5] 'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world prodlem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details)
        """
        act = Node(problem.actions[0])
        frontier = FIFOQueue()
        frontier.append(act)
        while(True):
            if not frontier:
                return None
            plan = frontier.pop()
            print(plan.state.name)
            hla = plan.state  # first_or_null(plan)
            prefix = None
            if plan.parent:
                prefix = plan.parent.state.action  # prefix, suffix = subseq(plan.state, hla)
            outcome = Problem.result(problem, prefix)
            if hla is None:
                if outcome.goal_test():
                    return plan.path()
            else:
                print("else")
                for sequence in Problem.refinements(hla, outcome, hierarchy):
                    print("...")
                    frontier.append(Node(plan.state, plan.parent, sequence))

    def result(problem, action):
        """The outcome of applying an action to the current problem"""
        if action is not None:
            problem.act(action)
            return problem
        else:
            return problem


def job_shop_problem():
    """
    [figure 11.1] JOB-SHOP-PROBLEM

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
    init = [expr('Car(C1)'),
            expr('Car(C2)'),
            expr('Wheels(W1)'),
            expr('Wheels(W2)'),
            expr('Engine(E2)'),
            expr('Engine(E2)')]

    def goal_test(kb):
        # print(kb.clauses)
        required = [expr('Has(C1, W1)'), expr('Has(C1, E1)'), expr('Inspected(C1)'),
                    expr('Has(C2, W2)'), expr('Has(C2, E2)'), expr('Inspected(C2)')]
        for q in required:
            # print(q)
            # print(kb.ask(q))
            if kb.ask(q) is False:
                return False
        return True

    resources = {'EngineHoists': 1, 'WheelStations': 2, 'Inspectors': 2, 'LugNuts': 500}

    # AddEngine1
    precond_pos = []
    precond_neg = [expr("Has(C1,E1)")]
    effect_add = [expr("Has(C1,E1)")]
    effect_rem = []
    add_engine1 = HLA(expr("AddEngine1"),
                      [precond_pos, precond_neg], [effect_add, effect_rem],
                      duration=30, use={'EngineHoists': 1})

    # AddEngine2
    precond_pos = []
    precond_neg = [expr("Has(C2,E2)")]
    effect_add = [expr("Has(C2,E2)")]
    effect_rem = []
    add_engine2 = HLA(expr("AddEngine2"),
                      [precond_pos, precond_neg], [effect_add, effect_rem],
                      duration=60, use={'EngineHoists': 1})

    # AddWheels1
    precond_pos = []
    precond_neg = [expr("Has(C1,W1)")]
    effect_add = [expr("Has(C1,W1)")]
    effect_rem = []
    add_wheels1 = HLA(expr("AddWheels1"),
                      [precond_pos, precond_neg], [effect_add, effect_rem],
                      duration=30, consume={'LugNuts': 20}, use={'WheelStations': 1})

    # AddWheels2
    precond_pos = []
    precond_neg = [expr("Has(C2,W2)")]
    effect_add = [expr("Has(C2,W2)")]
    effect_rem = []
    add_wheels2 = HLA(expr("AddWheels2"),
                      [precond_pos, precond_neg], [effect_add, effect_rem],
                      duration=15, consume={'LugNuts': 20}, use={'WheelStations': 1})

    # Inspect1
    precond_pos = []
    precond_neg = [expr("Inspected(C1)")]
    effect_add = [expr("Inspected(C1)")]
    effect_rem = []
    inspect1 = HLA(expr("Inspect1"),
                   [precond_pos, precond_neg], [effect_add, effect_rem],
                   duration=10, use={'Inspectors': 1})

    # Inspect2
    precond_pos = []
    precond_neg = [expr("Inspected(C2)")]
    effect_add = [expr("Inspected(C2)")]
    effect_rem = []
    inspect2 = HLA(expr("Inspect2"),
                   [precond_pos, precond_neg], [effect_add, effect_rem],
                   duration=10, use={'Inspectors': 1})

    job_group1 = [add_engine1, add_wheels1, inspect1]
    job_group2 = [add_engine2, add_wheels2, inspect2]

    return Problem(init, [add_engine1, add_engine2, add_wheels1, add_wheels2, inspect1, inspect2],
                   goal_test, [job_group1, job_group2], resources)
