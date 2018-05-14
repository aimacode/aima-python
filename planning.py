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
        precond_neg, precond_pos = partition(preconds, is_negative_clause)
        self.precond_pos = set(precond_pos)
        self.precond_neg = set(e.args[0] for e in precond_neg)  # change the negative Exprs to positive
        self.effects = effects
        effect_rem, effect_add = partition(effects, is_negative_clause)
        self.effect_add = set(effect_add)
        self.effect_rem = set(e.args[0] for e in effect_rem)  # change the negative Exprs to positive

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
        act.precond_pos = self.precond_pos.copy()
        act.precond_neg = self.precond_neg.copy()
        act.effects = self.effects.copy()
        act.effect_add = self.effect_add.copy()
        act.effect_rem = self.effect_rem.copy()
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
        yield from self.check_neg_precond(kb, self.precond_neg, self.check_pos_precond(kb, self.precond_pos, {}))

    def act(self, subst, kb):
        """ Executes the action on a new copy of the PlanningKB """
        new_kb = PlanningKB(kb.goal_clauses, kb.clause_set)
        clause_set = set(new_kb.clause_set)
        neg_literals = set(self.substitute(subst, clause) for clause in self.effect_rem)
        pos_literals = set(self.substitute(subst, clause) for clause in self.effect_add)
        new_kb.clause_set = frozenset(clause_set - neg_literals | pos_literals)
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
    domain_entries = [de for de in os.scandir(os.getcwd() + os.sep + 'pddl_files') if de.name.endswith('domain.pddl')]
    problem_entries = [de for de in os.scandir(os.getcwd() + os.sep + 'pddl_files') if de.name.endswith('problem.pddl')]
    domain_objects = []
    problem_objects = []
    for de in domain_entries:
        domain_parser = DomainParser()
        domain_parser.read(de.path)
        domain_objects.append(domain_parser)

    for de in problem_entries:
        problem_parser = ProblemParser()
        problem_parser.read(de.path)
        problem_objects.append(problem_parser)

    object_pairs = []
    for p in problem_objects:
        for d in domain_objects:
            if p.domain_name == d.domain_name:
                object_pairs.append((d, p))
    if object_pairs:
        return object_pairs
    else:
        raise ParseError('No matching PDDL domain and problem files found.')


def test_solutions():
    for domain, problem in gather_test_pairs():
        construct_solution_from_pddl(domain, problem)


if __name__ == '__main__':
    test_solutions()
