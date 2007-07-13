"""Probability models. (Chapter 13-15)
"""

from utils import *
from logic import extend
import agents
from random import random, seed

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
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> [P['lo'], P['med'], P['hi']]
    [0.125, 0.375, 0.5]
    >>> P = ProbDist('X', {'lo': 0.125, 'med': 0.250, 'hi': 0.625})
    >>> [P['lo'], P['med'], P['hi']]
    [0.125, 0.25, 0.625]
    """
    def __init__(self, varname='?', freqs=None):
        """If freqs is given, it is a dictionary of value: frequency pairs,
        and the ProbDist is normalized."""
        update(self, prob={}, varname=varname, values=[])
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        "Given a value, return P(value)."
        return self.prob[val]

    def __setitem__(self, val, p):
        "Set P(val) = p"
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.

        Returns the normalized distribution.

        Raises a ZeroDivisionError if the sum of the values is 0.
        
        >>> P = ProbDist('Flip'); P['H'], P['T'] = 35, 65
        >>> P = P.normalize()
        >>> print '%5.3f %5.3f' % (P.prob['H'], P.prob['T'])
        0.350 0.650
        """
        total = float(sum(self.prob.values()))
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
    Works for Boolean variables only. [Fig. 13.4]. *** or discrete only? ***

    X is a string (variable name).
    e is a dictionary of variable-name value pairs.
    P is an instance of JointProbDist."""
    
    Q = ProbDist(X) ## A probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e] # hidden variables
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
        # extend(e, X, xi) copies dictionary e and adds the pair X: xi
        # (from logic.py)
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

    """Bayesian network containing only boolean variable nodes."""
    
    def __init__(self, nodes=[]):
        update(self, nodes=[], vars=[])
        for node in nodes:
            self.add(node)

    def add(self, node):
        self.nodes.append(node)
        self.vars.append(node.variable)

    def observe(self, var, val):
        self.evidence[var] = val

    def variable_node (self, var):
        """Returns the node for the variable named var.

        >>> burglary.variable_node('Burglary').variable
        'Burglary'
        """
        
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)

    def variables (self):
        """Returns the list of names of the variables.

        >>> burglary.variables()
        ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']
        """
        
        return [n.variable for n in self.nodes]
    
    def variable_values (self, var):
        return [True, False]
        

class BayesNode:
    def __init__(self, variable, parents, cpt):
        if isinstance(parents, str): parents = parents.split()
        update(self, variable=variable, parents=parents, cpt=cpt)

node = BayesNode


# Burglary example [Fig. 14.2]

T, F = True, False

burglary = BayesNet([
    # It seems important in enumerate_all that variables (nodes)
    # be listed in such an order that parents precede all of their children.
    node('Burglary', '', .001), 
    node('Earthquake', '', .002),
    node('Alarm', 'Burglary Earthquake',
         { (T, T):.95,
           (T, F):.94,
           (F, T):.29,
           (F, F):.001}),
    node('JohnCalls', 'Alarm', {T:.90, F:.05}),
    node('MaryCalls', 'Alarm', {T:.70, F:.01})
    ])

#______________________________________________________________________________

def enumeration_ask (X, e, bn):
    """Returns a distribution of X given e from bayes net bn.  [Fig. 14.9]
    
    X is a string (variable name).
    e is a dictionary of variablename: value pairs.
    bn is an instance of BayesNet.
    
    >>> p = enumeration_ask('Earthquake', {}, burglary)
    >>> [p[True], p[False]]
    [0.002, 0.998]
    >>> p = enumeration_ask('Burglary',
    ...   {'JohnCalls': True, 'MaryCalls': True}, burglary)
    >>> [p[True], p[False]]
    [0.28417183536439289, 0.71582816463560706]
    """

    Q = ProbDist(X) # empty probability distribution for X
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables(), extend(e, X, xi), bn)
        # Assume that parents precede children in bn.variables,
        # otherwise in enumerate_all, the values of y's parents
        # may be unspecified.
    return Q.normalize()

def enumerate_all (vars, e, bn):
    """Returns a real number = ??? the probability that X = xi given e.

    vars is a list of variables in bn.
    e is a dictionary of variablename: value pairs
    bn is an instance of BayesNet.

    Precondition: no variable in vars precedes its parents."""
    if not vars: # i.e. []
        return 1.0
    else:
        Y = vars[0]
        rest = vars[1:]

        Ynode = bn.variable_node(Y)
        parents = Ynode.parents
        cpt = Ynode.cpt
        
        if e.has_key(Y):
            y = e[Y]
            cp = condprob(cpt, y, parents, e) # P(y | parents(Y))
            result = cp * enumerate_all(rest, e, bn)
        else:
            result = 0
            for y in bn.variable_values(Y):
                cp = condprob(cpt, y, parents, e) # P(y | parents(Y)
                result += cp * enumerate_all(rest, extend(e, Y, y), bn)

        return result

def condprob (cpt, y, parent_vars, evidence):
    """Return the conditional probability P(y | parent_vars = parent_values)
    by lookup in cpt, where parent_values are the values that
    parent_vars have in evidence.

    cpt is a conditional probability table for a variable Y:
    a dictionary if Y has parents; otherwise a single number.
    y is a possible value of a boolean random variable Y.
    parent_vars is a tuple of the names of the parents of Y.
    evidence is a dictionary of variablename: value pairs.

    Preconditions:
    1.  each variable in parent_vars is bound to a value in evidence.
    2.  the variables are listed in parent_vars in the same order
    in which they are listed in cpt.

    >>> cpt = burglary.variable_node('Alarm').cpt
    >>> parents = ['Burglary', 'Earthquake']
    >>> evidence = {'Burglary': True, 'Earthquake': True}
    >>> print '%4.2f' % condprob(cpt, True, parents, evidence)
    0.95
    >>> evidence = {'Burglary': False, 'Earthquake': True}
    >>> print '%4.2f' % condprob(cpt, False, parents, evidence)
    0.71
    >>> print '%4.2f' % condprob(0.75, False, [], {})
    0.25
    """

    # It is a little unpleasant that while enumeration_ask and
    # enumerate_all are independent of the variables being boolean,
    # this is not.
    # Can that assumption be confined to the BayesNet class?
    # Or to a particular CPT class, BooleanCPT?
    # Also that cpt may be either a dictionary or a number.

    if parent_vars == []:
        py = cpt
    else:
        parent_values = [evidence[parent] for parent in parent_vars]
        if len(parent_values) == 1:
            key = parent_values[0]
        else:
            key = tuple(parent_values)
        py = cpt[key]
    if y:
        return py # P(Y = True)
    else:
        return 1.0 - py # P(Y = False)


def condsamp (cpt, parent_vars, evidence):
    
    """Return a sample value True or False from the conditional
    distribution cpt of an unspecified variable, given its parent
    parent_vars have the values they have in evidence

    >>> cpt = {True: 0.2, False: 0.7}
    >>> condsamp(cpt, ['A'], {'A': True}) in [True, False]
    True
    >>> cpt = {(True, True): 0.1, (True, False): 0.3,
    ...   (False, True): 0.5, (False, False): 0.7}
    >>> condsamp(cpt, ['A', 'B'], {'A': True, 'B': False}) in [True, False]
    True
    """
    
    p = condprob(cpt, True, parent_vars, evidence)
    return (random() <= p)

#______________________________________________________________________________

# elimination_ask: implementation is incomplete

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

# Fig. 14.11a: sprinkler network

sprinkler = BayesNet([
    node('Cloudy', '', 0.5),
    node('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
    node('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
    node('WetGrass', 'Sprinkler Rain',
         { (T, T): 0.99,
           (T, F): 0.90,
           (F, T): 0.90,
           (F, F): 0.00})
    ])

#______________________________________________________________________________

def prior_sample(bn):
    """[Fig. 14.12]

    Argument: bn is an instance of BayesNet.
    Returns: one sample, a dictionary of variable-name: value pairs.

    >>> s = prior_sample(burglary)
    >>> s['Burglary'] in [True, False]
    True
    >>> s['Alarm'] in [True, False]
    True
    >>> s['JohnCalls'] in [True, False]
    True
    >>> len(s)
    5
    """
    sample = {} # boldface x in Fig. 14.12
    for node in bn.nodes:
        var = node.variable
        sample[var] = condsamp(node.cpt, node.parents, sample)
    return sample

#_______________________________________________________________________________

def rejection_sampling (X, e, bn, N):
    """Estimates probability distribution of X given evidence e
    in BayesNet bn, using N samples.  [Fig. 14.13]

    Arguments:
    X is a variable name (string).
    e is a dictionary of variable-name: value pairs.
    bn is an instance of BayesNet.
    N is an integer > 0.

    Returns: an instance of ProbDist representing P(X | e).

    Raises a ZeroDivisionError if all the N samples are rejected,
    i.e., inconsistent with e.

    >>> seed(21); p = rejection_sampling('Earthquake', {}, burglary, 1000)
    >>> [p[True], p[False]]
    [0.001, 0.999]
    >>> seed(47)
    >>> p = rejection_sampling('Burglary',
    ...   {'JohnCalls': True, 'MaryCalls': True}, burglary, 10000)
    >>> [p[True], p[False]]
    [0.29999999999999999, 0.69999999999999996]
    """

    counts = {True: 0, False: 0} # counts is boldface N in Fig. 14.13

    # Generate and count observations
    for j in xrange(N):
        sample = prior_sample(bn) # sample is boldface x in Fig. 14.13
        if consistent_with(sample, e):
            counts[sample[X]] += 1 # increment count of sampled value of X
            
    # Package counts as a ProbDist
    d = ProbDist(X)
    for value in bn.variable_values(X):
        d[value] = counts[value]
    return d.normalize()

def consistent_with (sample, evidence):
    """Returns True if sample is consistent with evidence, False otherwise.

    sample is a dictionary of variable-name: value pairs.
    evidence is a dictionary of variable-name: value pairs.
    The variable names in evidence are a subset of the variable names
    in sample.

    >>> s = {'A': True, 'B': False, 'C': True, 'D': False}
    >>> consistent_with(s, {})
    True
    >>> consistent_with(s, s)
    True
    >>> consistent_with(s, {'A': False})
    False
    >>> consistent_with(s, {'D': True})
    False
    """
    for (k, v) in evidence.items():
        if sample[k] != v:
            return False
    return True

#_______________________________________________________________________________

# Fig. 14.14: likelihood_weighting, weighted_sample

def likelihood_weighting (X, e, bn, N):
    """Returns an estimate of P(X | e).

    Arguments:
    X is a variable name (string).
    e is a dictionary of variable-name: value pairs (the evidence).
    bn is an instance of BayesNet.
    N is an integer, the number of samples to be generated.

    Returns an instance of ProbDist.
    
    >>> seed(71); p = likelihood_weighting('Earthquake', {}, burglary, 1000)
    >>> [p[True], p[False]]
    [0.002, 0.998]
    >>> seed(1017)
    >>> p = likelihood_weighting('Burglary',
    ...  {'JohnCalls': True, 'MaryCalls': True}, burglary, 10000)
    >>> [p[True], p[False]]
    [0.29801552320954111, 0.70198447679045894]
    """

    # Initialize weighted counts of X values
    weights = {True: 0.0, False: 0.0} # boldface W in Fig. 14.14

    # Generate and count observations
    for j in xrange(N):
        sample, weight = weighted_sample(bn, e) # boldface x, w in Fig. 14.14
        sample_X = sample[X] # value of X in sample
        weights[sample_X] += weight

    # Package and return weights as a ProbDist
    return ProbDist(X, weights)
    
def weighted_sample (bn, e):
    """Returns an event (a sample) and a weight."""

    # Initialize
    event = {} # will store variables and values (boldface x in Fig. 14.14)
    weight = 1.0 # w in Fig. 14.14

    # Accumulate event and weight
    for node in bn.nodes:
        X = node.variable # X sub i in Fig. 14.14
        parents = node.parents
        cpt = node.cpt
#        print "Variable %s, parents %s, cpt %s" % (X, parents, cpt)
        if e.has_key(X):
            xvalue = e[X]
            event[X] = xvalue
            cp = condprob(cpt, xvalue, parents, event)
            weight *= cp
#            print "    value = %s, cp = %f, weight = %f" % (xvalue, cp, weight)
            
        else:
            event[X] = condsamp(cpt, parents, event)
#            print "    value = %s" % event[X]
        
    return event, weight
    

#_______________________________________________________________________________

# MISSING

# Fig. 14.15: mcmc_ask

