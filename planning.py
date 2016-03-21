"""Planning (Chapters 10-11)
"""
from logic import *
from utils import *
import agents
from search import *

import math
import random
import sys
import time
import bisect
import string


class Action(Expr):
    """Subclasses Expr class to define an Action class that can refer
    to actions of an existing Node or a HLA"""
    def __init__(self,*args):
        self.args = list(map(expr, str(args)))  # Coerce args to Exprs
        
    def Refinements(self,outcome,frontier):
        """Refinement function on HLA using frontier and outcome
        Steps = frontier    Precondition = outcome"""
        X = Node(self).path()
        R = []
        for each in X:
            if each.state == outcome:
                R.append(each.action)        
        return R
              
#_____________________________________________________________________________#

    
def hierarchicalSearch(problem):
    """ Artificial Intelligence A Modern Approach Figure 11.5 function
    HIERARCHICAL_SEARCH(problem,hierarchy) returns a solution or failure                  
    frontier <- a FIFO queue with[Act] as the only element
    loop do
        if EMPTY?(frontier)then return failure
        plan<-POP(frontier) //chooses the shallowest plan in frontier
        hla<-the first HLA in plan,or null if none
        prefix,suffix <-action before and after hla in plan
        outcome <- RESULT(problem.INITIAL-STATE,prefix)
        if hla is null then //so plan is primitive
            if outcome statisfies.problem.GOAL then return plan
        else for each sequence in REFINEMENTS(hla,outcome,hierarchy)do
            frontier <-INSERT(APPEND(prefix,sequence,suffix),frontier)
    Figure 11.5 A breadth-first implementation of hierarchical forward
    planning search.The initial plan supplied to the algorithm is [Act]. 
    The REFINEMENTS function returns a set of action sequences, one for 
    each refinement of the HLA whose preconditions are satisfied by the
    specified state, outcome.
    author :Sarthak Mahapatra"""

    current = Node(problem.initial)
    frontier = FIFOQueue()
    frontier.append(current)
    while True:
        if not frontier:
            return None
        plan = frontier.pop()
        hla = Action(plan.action)
        prefix = Action(plan.parent.action)
        suffix = Action(Node.child_node(plan,problem,hla))
        outcome = problem.result(problem.initial,prefix)
        if hla is None:
            if problem.goal_test(outcome):
                return success(plan)
        else:
            actionSequence = hla.Refinements(outcome,frontier)
            actionSequence.append(prefix)
            actionSequence.append(suffix)
            for actions in actionSequence :
                n = Node.__init__(plan.state, plan.parent, actions, plan.path_cost)
                frontier.append(n)
            
#_____________________________________________________________________________#


def success(plan):
    """Returns a list appended with the sucessful action"""
    k = []
    k.append(plan.action)
    return k
    
#_____________________________________________________________________________#





