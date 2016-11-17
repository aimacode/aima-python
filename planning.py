"""Planning (Chapters 10-11)
"""
import copy
from logic import fol_bc_and
from utils import expr, Expr, partition
from search import Problem, astar_search


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

        # heuristic is just whether remaining unresolved goals in the current KB are less than the remaining unsolved
        # goals in the other KB.
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


class PlanningProblem(Problem):
    """
    Used to define a planning problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely define a state.
    """
    def __init__(self, initial_state, actions, goals):
        super().__init__(initial_state, goals)
        self.action_list = actions

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.initial, self.action_list, self.goal)

    def actions(self, state):
        for action in self.action_list:
            for subst in action.check_precond(state):
                new_action = copy.deepcopy(action)
                new_action.subst = subst
                yield new_action

    def goal_test(self, state):
        return state.goal_test()

    def result(self, state, action):
        return action.act(action.subst, state)

    def h(self, node):
        return node.state.h()

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


class Action:
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


def print_solution(node):
    for action in node.solution():
        print(action.name, end='(')
        for a in action.args[:-1]:
            print('{},'.format(action.subst.get(a, a)), end=' ')
        print('{})'.format(action.subst.get(action.args[-1], action.args[-1])))


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

    p = PlanningProblem(init, [load, unload, fly], goals)
    n = astar_search(p)
    print_solution(n)


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

    p = PlanningProblem(init, [remove_spare, remove_flat, put_on_spare, leave_overnight], goals)
    n = astar_search(p)
    print_solution(n)


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

    p = PlanningProblem(init, [move, move_to_table], goals)
    n = astar_search(p)
    print_solution(n)


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

    p = PlanningProblem(init, [move, move_to_table], goals)
    n = astar_search(p)
    print_solution(n)


if __name__ == '__main__':
    air_cargo()
    print()
    spare_tire()
    print()
    three_block_tower()
    print()
    sussman_anomaly()
