"""Planning (Chapters 10-11)
"""
from logic import Expr
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
    def isNull(self):
        """Returns True if the HLA is Empty"""
        if self == None:
            return True
        else:
            return False
        
    def Refinements(self,outcome,frontier):
        """Refinement function on HLA using frontier and outcome
        Steps = frontier    Precondition = outcome"""
        X=Node(self).path()
        R = []
        for each in X:
            if each.state == outcome:
                R.append(each.action)        
        for each in R:
            R.append(prefix)
            R.append(suffix)
            for each in R:
                n = Node.__init__(plan,state, plan.parent,plan.action)
                frontier.append(n)
              
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
            return failure()
        plan = frontier.pop()
        hla = Action(plan.action)
        prefix = Action(plan.parent.action)
        suffix = Action(Node.child_node(plan,problem,hla))
        outcome = problem.result(problem.initial,prefix)
        if hla.isNull():
            if problem.goal_test(outcome):
                return success(plan)
        else:
            hla.Refinements(outcome,frontier)
            
#_____________________________________________________________________________#


def success(plan):
    """Returns a list appended with the sucessful action"""
    k = []
    k.append(plan.action)
    return k

def failure():
    """Returns an empty list to signify failure"""
    k = []
    return k


#_____________________________________________________________________________#


# Simplified road map of Romania
Fig[3, 2] = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
Fig[3, 2].locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))


