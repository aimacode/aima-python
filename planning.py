"""Planning (Chapters 10-11)
"""

from utils import Expr, expr, first
from logic import FolKB


class PDLL:
    """
    PDLL used to deine a search problem
    It stores states in a knowledge base consisting of first order logic statements
    The conjunction of these logical statements completely define a state
    """

    def __init__(self, initial_state, actions, goal_test):
        self.kb = FolKB(initial_state)
        self.actions = actions
        self.goal_test_func = goal_test

    def goal_test(self):
        return self.goal_test_func(self.kb)

    def act(self, action):
        """
        Performs the action given as argument
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
    Defines an action schema using preconditions and effects
    Use this to describe actions in PDDL
    action is an Expr where variables are given as arguments(args)
    Precondition and effect are both lists with positive and negated literals
    Example:
    precond_pos = [expr("Human(person)"), expr("Hungry(Person)")]
    precond_neg = [expr("Eaten(food)")]
    effect_add = [expr("Eaten(food)")]
    effect_rem = [expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), [precond_pos, precond_neg], [effect_add, effect_rem])
    """

    def __init__(self, action, precond, effect):
        self.name = action.op
        self.args = action.args
        self.precond_pos = precond[0]
        self.precond_neg = precond[1]
        self.effect_add = effect[0]
        self.effect_rem = effect[1]

    def __call__(self, kb, args):
        return self.act(kb, args)

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propostional symbol"""
        new_args = [args[i] for x in e.args for i in range(len(self.args)) if self.args[i] == x]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""
        # check for positive clauses
        for clause in self.precond_pos:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        # check for negative clauses
        for clause in self.precond_neg:
            if self.substitute(clause, args) in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's kb"""
        # check if the preconditions are satisfied
        if not self.check_precond(kb, args):
            raise Exception("Action pre-conditions not satisfied")
        # remove negative literals
        for clause in self.effect_rem:
            kb.retract(self.substitute(clause, args))
        # add positive literals
        for clause in self.effect_add:
            kb.tell(self.substitute(clause, args))


def air_cargo():
    init = [expr('At(C1, SFO)'),
            expr('At(C2, JFK)'),
            expr('At(P1, SFO)'),
            expr('At(P2, JFK)'),
            expr('Cargo(C1)'),
            expr('Cargo(C2)'),
            expr('Plane(P1)'),
            expr('Plane(P2)'),
            expr('Airport(JFK)'),
            expr('Airport(SFO)')]

    def goal_test(kb):
        required = [expr('At(C1 , JFK)'), expr('At(C2 ,SFO)')]
        for q in required:
            if kb.ask(q) is False:
                return False
        return True

    ## Actions
    #  Load
    precond_pos = [expr("At(c, a)"), expr("At(p, a)"), expr("Cargo(c)"), expr("Plane(p)"), expr("Airport(a)")]
    precond_neg = []
    effect_add = [expr("In(c, p)")]
    effect_rem = [expr("At(c, a)")]
    load = Action(expr("Load(c, p, a)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #  Unload
    precond_pos = [expr("In(c, p)"), expr("At(p, a)"), expr("Cargo(c)"), expr("Plane(p)"), expr("Airport(a)")]
    precond_neg = []
    effect_add = [expr("At(c, a)")]
    effect_rem = [expr("In(c, p)")]
    unload = Action(expr("Unload(c, p, a)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #  Load
    #  Used used 'f' instead of 'from' because 'from' is a python keyword and expr uses eval() function
    precond_pos = [expr("At(p, f)"), expr("Plane(p)"), expr("Airport(f)"), expr("Airport(to)")]
    precond_neg = []
    effect_add = [expr("At(p, to)")]
    effect_rem = [expr("At(p, f)")]
    fly = Action(expr("Fly(p, f, to)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDLL(init, [load, unload, fly], goal_test)
