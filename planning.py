"""Planning (Chapters 10-11)
"""

from utils import *
import agents
from search import Node

import math
import random
import sys
import time
import bisect
import string
from test.test_asyncio.test_events import noop


def hierarchicalSearch(problem):
    """ Artificial Intelligence A Modern Approach (3rd Edition): Figure 11.5
 function HIERARCHICAL_SEARCH(problem,hierarchy) returns a solution or failure                  
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
    Figure 11.5 A breadth-first implementation of hierarchical forward planning search.
    The initial plan supplied to the algorithm is [Act]. The REFINEMENTS function 
    returns a set of action sequences, one for each refinement of the HLA whose preconditions
    are satisfied by the specified state, outcome.
    author :Sarthak Mahapatra"""

    current=Node(problem.initial)
    frontier=FIFOQueue()
    frontier.append(current)
    while True:
        if frontier.__len__()==0:
            return failure()
        plan=frontier.pop()
        hla=plan.action
        prefix=plan.parent.action
        suffix=Node.child_node(plan,problem,hla)
        outcome=problem.result(problem.initial,prefix)
        if hla==None:
            if problem.goal_test(outcome):
                return success(plan)
        else:
            x=Refinements(hla,outcome,frontier)
            x.append(prefix)
            x.append(suffix)
            for each in x:
                n=Node.__init__(plan,state, plan.parent,plan.action)
                frontier.append(n)

def Refinements(hla,outcome,frontier):
    X=Node(hla).path()
    R=[]
    for each in X:
        if each.state==outcome:
            R.append(each.action)
    return R

def success(plan):
    k=[]
    k.append(plan.action)
    return k

def failure():
    k=[]
    return k



