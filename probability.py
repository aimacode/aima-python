"""Probability models. (Chapter 13-15)
"""

from utils import *
from logic import extend
import agents
import bisect, random

#______________________________________________________________________________

class DTAgent(agents.Agent):
    "A decision-theoretic agent. [Fig. 13.1]"

    def __init__(self, belief_state):
        agents.Agent.__init__(self)

        def program(percept):
            belief_state.observe(action, percept)
            program.action = argmax(belief_state.actions(), 
                                    belief_state.expected_outcome_utility)
            return program.action

        program.action = None
        self.program = program

#______________________________________________________________________________

class ProbDist:
    """A discrete probability distribution.  You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.5, 0.5; P['H']
    0.5
    """
    def __init__(self, varname='?'):
        update(self, prob={}, varname=varname, values=[])

    def __getitem__(self, val):
        "Given a value, return P(value)."
        return self.prob[val]

    def __setitem__(self, val, p):
        "Set P(val) = p"
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        "Make sure the probabilities of all values sum to 1."
        total = sum(self.prob.values())
        if not (1.0-epsilon < total < 1.0+epsilon):
            for val in self.prob:
                self.prob[val] /= total
        return self

epsilon = 0.001

class JointProbDist(ProbDist):
    """A discrete probability distribute over a set of variables.
    >>> P = JointProbDist(['X', 'Y']); P[1, 1] = 0.25
    >>> P[1, 1]
    0.25
    """
    def __init__(self, variables):
        update(self, prob={}, variables=variables, vals=DefaultDict([]))

    def __getitem__(self, values):
        "Given a tuple or dict of values, return P(values)."
        if isinstance(values, dict):
            values = tuple([values[var] for var in self.variables])
        return self.prob[values]

    def __setitem__(self, values, p):
        """Set P(values) = p.  Values can be a tuple or a dict; it must
        have a value for each of the variables in the joint. Also keep track
        of the values we have seen so far for each variable."""
        if isinstance(values, dict):
            values = [values[var] for var in self.variables]
        self.prob[values] = p
        for var,val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        "Return the set of possible values for a variable."
        return self.vals[var]

    def __repr__(self):
        return "P(%s)" % self.variables

#______________________________________________________________________________

def enumerate_joint_ask(X, e, P):
    """Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P. 
    Works for Boolean variables only. [Fig. 13.4]"""
    Q = ProbDist(X) ## A probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e]
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()

def enumerate_joint(vars, values, P):
    "As in Fig 13.4, except x and e are already incorporated in values."
    if not vars: 
        return P[values] 
    Y = vars[0]; rest = vars[1:]
    return sum([enumerate_joint(rest, extend(values, Y, y), P) 
                for y in P.values(Y)])

#______________________________________________________________________________

class BayesNet:
    def __init__(self, nodes=[]):
        update(self, nodes=[], vars=[])
        for node in nodes:
            self.add(node)

    def add(self, node):
        self.nodes.append(node)
        self.vars.append(node.variable)

    def observe(self, var, val):
        self.evidence[var] = val

class BayesNode:
    def __init__(self, variable, parents, cpt):
        if isinstance(parents, str): parents = parents.split()
        update(self, variable=variable, parents=parents, cpt=cpt)

node = BayesNode


T, F = True, False

burglary = BayesNet([
  node('Burglary', '', .001), 
  node('Earthquake', '', .002),
  node('Alarm', 'Burglary Earthquake', {
                    (T, T):.95,
                    (T, F):.94,
                    (F, T):.29,
                    (F, F):.001}),
  node('JohnCalls', 'Alarm', {T:.90, F:.05}),
  node('MaryCalls', 'Alarm', {T:.70, F:.01})
  ])
#______________________________________________________________________________

def elimination_ask(X, e, bn):
    "[Fig. 14.10]"
    factors = []
    for var in reverse(bn.vars):
        factors.append(Factor(var, e)) 
        if is_hidden(var, X, e):
            factors = sum_out(var, factors)
    return pointwise_product(factors).normalize()

def pointwise_product(factors):
    pass

def sum_out(var, factors):
    pass

#______________________________________________________________________________

def prior_sample(bn):
    x = {}
    for xi in bn.vars:
        x[xi.var] = xi.sample([x])
    
#______________________________________________________________________________        

