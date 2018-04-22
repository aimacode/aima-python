"""Planning (Chapters 10-11)
"""
import os
from logic import fol_bc_and
from utils import expr, Expr, partition
from search import astar_search
from pddl_parse import DomainParser, ProblemParser, ParseError


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


class PlanningProblem:
    """
    Used to define a planning problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely define a state.
    """
    def __init__(self, initial_kb, actions):
        self.initial = initial_kb
        self.possible_actions = actions

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.initial, self.possible_actions)

    def actions(self, state):
        for action in self.possible_actions:
            for subst in action.check_precond(state):
                new_action = action.copy()
                new_action.subst = subst
                yield new_action

    def goal_test(self, state):
        return state.goal_test()

    def result(self, state, action):
        return action.act(action.subst, state)

    def h(self, node):
        return node.state.h()

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


def is_negative_clause(e):
    return e.op == '~' and len(e.args) == 1


class PlanningAction:
    """
    Defines an action schema using preconditions and effects
    Use this to describe actions in PDDL
    action is an Expr where variables are given as arguments(args)
    Precondition and effect are both lists with positive and negated literals
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("~Eaten(food)")]
    effect = [expr("Eaten(food)"), expr("~Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, expression, preconds, effects):
        self.expression = expression
        self.name = expression.op
        self.args = expression.args
        self.subst = None
        self.preconds = preconds
        self.effects = effects

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, Expr(self.name, *self.args),
                                       list(self.preconds), list(self.effects))

    def copy(self):
        """ Returns a copy of this object. """
        act = self.__new__(self.__class__)
        act.name = self.name
        act.args = self.args[:]
        act.subst = self.subst
        act.preconds = self.preconds.copy()
        act.effects = self.effects.copy()
        return act

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
        precond_neg, precond_pos = partition(self.preconds, is_negative_clause)
        precond_neg = set(e.args[0] for e in precond_neg)  # change the negative Exprs to positive
        yield from self.check_neg_precond(kb, precond_neg, self.check_pos_precond(kb, precond_pos, {}))

    def act(self, subst, kb):
        """ Executes the action on a new copy of the PlanningKB """
        new_kb = PlanningKB(kb.goal_clauses, kb.clause_set)
        neg_effects, pos_effects = partition(self.effects, is_negative_clause)
        neg_effects = set(self.substitute(subst, e.args[0]) for e in neg_effects)
        pos_effects = set(self.substitute(subst, e) for e in pos_effects)
        new_kb.clause_set = frozenset(kb.clause_set - neg_effects | pos_effects)
        return new_kb


def print_solution(node):
    for action in node.solution():
        print(action.name, end='(')
        for a in action.args[:-1]:
            print('{},'.format(action.subst.get(a, a)), end=' ')
        if action.args:
            print('{})'.format(action.subst.get(action.args[-1], action.args[-1])))
        else:
            print(')')
    print()


def construct_solution_from_pddl(pddl_domain, pddl_problem) -> None:
    initial_kb = PlanningKB([expr(g) for g in pddl_problem.goals],
                            [expr(s) for s in pddl_problem.initial_state])

    planning_actions = [PlanningAction(expr(name),
                                       [expr(p) for p in preconds],
                                       [expr(e) for e in effects])
                        for name, preconds, effects in pddl_domain.actions]
    p = PlanningProblem(initial_kb, planning_actions)
    print('\n{} solution:'.format(pddl_problem.problem_name))
    print_solution(astar_search(p))


def gather_test_pairs() -> list:
    pddl_direntries = os.scandir(os.getcwd() + os.sep + 'pddl_files')
    domain_objects = []
    problem_objects = []
    for de in pddl_direntries:
        try:
            domain_parser = DomainParser()
            domain_parser.read(de.path)
            domain_objects.append(domain_parser)
        except ParseError:
            try:
                problem_parser = ProblemParser()
                problem_parser.read(de.path)
                problem_objects.append(problem_parser)
            except ParseError:
                raise ParseError('Unparseable PDDL file: {}'.format(de.name))

    object_pairs = []
    for p in problem_objects:
        for d in domain_objects:
            if p.domain_name == d.domain_name:
                object_pairs.append((d, p))
    return object_pairs


def spare_tire():
    goals = [expr('At(Spare, Axle)')]
    init = PlanningKB(goals,
                      [expr('At(Flat, Axle)'),
                       expr('At(Spare, Trunk)')])
    # Actions
    #  Remove(Spare, Trunk)
    precond = [expr('At(Spare, Trunk)')]
    effect = [expr('At(Spare, Ground)'), expr('~At(Spare, Trunk)')]
    remove_spare = PlanningAction(expr('Remove(Spare, Trunk)'), precond, effect)
    #  Remove(Flat, Axle)
    precond = [expr('At(Flat, Axle)')]
    effect = [expr('At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    remove_flat = PlanningAction(expr('Remove(Flat, Axle)'), precond, effect)
    #  PutOn(Spare, Axle)
    precond = [expr('At(Spare, Ground)'), expr('~At(Flat, Axle)')]
    effect = [expr('At(Spare, Axle)'), expr('~At(Spare, Ground)')]
    put_on_spare = PlanningAction(expr('PutOn(Spare, Axle)'), precond, effect)
    #  LeaveOvernight
    precond = []
    effect = [expr('~At(Spare, Ground)'), expr('~At(Spare, Axle)'), expr('~At(Spare, Trunk)'),
              expr('~At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    leave_overnight = PlanningAction(expr('LeaveOvernight'), precond, effect)
    p = PlanningProblem(init, [remove_spare, remove_flat, put_on_spare, leave_overnight])
    print_solution(astar_search(p))


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
    move = PlanningAction(expr('Move(b, x, y)'), precond, effect)

    #  MoveToTable(b, x)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    effect = [expr('On(b, Table)'), expr('Clear(x)'), expr('~On(b, x)')]
    move_to_table = PlanningAction(expr('MoveToTable(b, x)'), precond, effect)

    p = PlanningProblem(init, [move, move_to_table])
    print_solution(astar_search(p))


def put_on_shoes():
    goals = [expr('On(RightShoe, RF)'), expr('On(LeftShoe, LF)')]
    init = PlanningKB(goals, [expr('Clear(LF)'),
                              expr('Clear(RF)'),
                              expr('LeftFoot(LF)'),
                              expr('RightFoot(RF)')])

    # Actions
    #  RightShoe
    precond = [expr('On(RightSock, x)'), expr('RightFoot(x)'), expr('~On(RightShoe, x)')]
    effect = [expr('On(RightShoe, x)')]
    right_shoe = PlanningAction(expr('RightShoeOn'), precond, effect)

    #  RightSock
    precond = [expr('Clear(x)'), expr('RightFoot(x)')]
    effect = [expr('On(RightSock, x)'), expr('~Clear(x)')]
    right_sock = PlanningAction(expr('RightSockOn'), precond, effect)

    #  LeftShoe
    precond = [expr('On(LeftSock, x)'), expr('LeftFoot(x)'), expr('~On(LeftShoe, x)')]
    effect = [expr('On(LeftShoe, x)')]
    left_shoe = PlanningAction(expr('LeftShoeOn'), precond, effect)

    #  LeftSock
    precond = [expr('Clear(x)'), expr('LeftFoot(x)')]
    effect = [expr('On(LeftSock, x)'), expr('~Clear(x)')]
    left_sock = PlanningAction(expr('LeftSockOn'), precond, effect)

    p = PlanningProblem(init, [right_shoe, right_sock, left_shoe, left_sock])
    print_solution(astar_search(p))


def test_solutions():
    for domain, problem in gather_test_pairs():
        construct_solution_from_pddl(domain, problem)


if __name__ == '__main__':
    test_solutions()
