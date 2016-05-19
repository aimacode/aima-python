"""Planning (Chapters 10-11)
"""

from utils import Expr
from logic import FolKB

class Action():
    """
    Defines an action schema using preconditions and effects
    Use this to describe actions in PDDL
    action is an Expr where variables are given as arguments(args)
    Precondition and effect are both lists with positive and negated literals
    eat = Action([expr("Hungry"), ], [, ])
    """

    def __init__(self,action , precond, effect):
        self.name = action.op
        self.args = action.args
        self.precond_pos = precond[0]
        self.precond_neg = precond[1]
        self.effect_pos = effect[0]
        self.effect_neg = effect[1]

    def __call__(self, kb, args):
        return self.act(kb, args)

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propostional symbol"""
        new_args = [args[i] for i in range(len(self.args)) for x in e.args if self.args[i]==x]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""
        #check for positive clauses
        for clause in self.precond_pos:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        #check for negative clauses
        for clause in self.precond_neg:
            if self.substitute(clause, args) in kb.clause:
                return False
        return True

    def act(self, kb):
        """Executes the action on the state's kb"""
        #check if the preconditions are satisfied
        if not self.check_precond(kb):
            raise Exception("Action pre-conditions not satisfied")
        #remove negative literals
        for clause in self.effect_neg:
            kb.retract(self.substitute(clause, args))
        #add positive literals
        for clause in self.effect_pos:
            kb.tell(self.substitute(clause, args))
