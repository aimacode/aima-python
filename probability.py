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
        and the ProbDist then is normalized."""
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

    def show_approx(self, ndigits=3):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join(['%s: %.*g' % (v, ndigits, p)
                          for (v, p) in sorted(self.prob.items())])

epsilon = 0.001

class JointProbDist(ProbDist):
    """A discrete probability distribute over a set of variables.
    >>> P = JointProbDist(['X', 'Y']); P[1, 1] = 0.25
    >>> P[1, 1]
    0.25
    >>> P[dict(X=0, Y=1)] = 0.5
    >>> P[dict(X=0, Y=1)]
    0.5"""
    def __init__(self, variables):
        update(self, prob={}, variables=variables, vals=DefaultDict([]))

    def __getitem__(self, values):
        "Given a tuple or dict of values, return P(values)."
        return self.prob[event_values(values, self.variables)]

    def __setitem__(self, values, p):
        """Set P(values) = p.  Values can be a tuple or a dict; it must
        have a value for each of the variables in the joint. Also keep track
        of the values we have seen so far for each variable."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        "Return the set of possible values for a variable."
        return self.vals[var]

    def __repr__(self):
        return "P(%s)" % self.variables


#______________________________________________________________________________


class BoolCpt:
    """Conditional probability table for a boolean (True/False)
    random variable conditioned on its parents."""

    def __init__(self, table_data):
        """Initialize the table.

        table_data may have one of three forms, depending on the
        number of parents:
        
        1.  If the variable has no parents, table_data MAY be
        a single number (float), representing P(X = True).

        2.  If the variable has one parent, table_data MAY be
        a dictionary containing items of the form v: p,
        where p is P(X = True | parent = v).
        
        3.  If the variable has n parents, n > 1, table_data MUST be
        a dictionary containing items (v1, ..., vn): p,
        where p is P(P = True | parent1 = v1, ..., parentn = vn).

        (Form 3 is also allowed in the case of zero or one parent.)

        >>> cpt = BoolCpt(0.2)
        >>> T = True; F = False
        >>> cpt = BoolCpt({T: 0.2, F: 0.7})
        >>> cpt = BoolCpt({(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        # A little work here makes looking up values MUCH simpler
        # later on.  We transform table_data into the standard form
        # of a dictionary {(value, ...): number, ...} even if
        # the tuple has just 0 or 1 value.
        if type(table_data) == float: # no parents, 0-tuple
            self.table_data = {(): table_data}
        elif type(table_data) == dict:
            keys = table_data.keys()
            if type(keys[0]) == bool: # one parent, 1-tuple
                d = {}
                for k in keys:
                    d[(k,)] = table_data[k]
                self.table_data = d
            elif type(keys[0]) == tuple: # normal case, n-tuple
                self.table_data = table_data
            else:
                raise Exception("wrong key type: %s" % table_data)
        else:
            raise Exception("wrong table_data type: %s" % table_data)
        
    def p(self, value, parent_vars, event):
        """Return the conditional probability P(value | parent_vars =
        parent_values), where parent_values are the values of
        parent_vars in event.

        value is True or False.
        parent_vars is a list or tuple of variable names (strings).
        event is a dictionary of variable-name: value pairs.

        Preconditions:
        1.  each variable in parent_vars is bound to a value in event.
        2.  the variables are listed in parent_vars in the same order
        in which they are listed in the Cpt.

        >>> cpt = burglary.variable_node('Alarm').cpt
        >>> parents = ['Burglary', 'Earthquake']
        >>> event = {'Burglary': True, 'Earthquake': True}
        >>> print '%4.2f' % cpt.p(True, parents, event)
        0.95
        >>> event = {'Burglary': False, 'Earthquake': True}
        >>> print '%4.2f' % cpt.p(False, parents, event)
        0.71
        >>> BoolCpt({T: 0.2, F: 0.625}).p(False, ['Burglary'], event)
        0.375
        >>> BoolCpt(0.75).p(False, [], {})
        0.25
        """
        return self.p_values(value, event_values(event, parent_vars))

    def p_values(self, xvalue, parent_values):
        """Return P(X = xvalue | parents = parent_values),
        where parent_values is a tuple, even if of only 0 or 1 element.
        >>> cpt = BoolCpt(0.25)
        >>> cpt.p_values(F, ())
        0.75
        >>> cpt = BoolCpt({T: 0.25, F: 0.625})
        >>> cpt.p_values(T, (T,))
        0.25
        >>> cpt.p_values(F, (F,))
        0.375
        >>> cpt = BoolCpt({(T, T): 0.2, (T, F): 0.31,
        ...  (F, T): 0.5, (F, F): 0.62})
        >>> cpt.p_values(T, (T, F))
        0.31
        >>> cpt.p_values(F, (F, F))
        0.38
        """
        ptrue = self.table_data[parent_values] # True or False
        if xvalue:
            return ptrue
        else:
            return 1.0 - ptrue

    def rand(self, parents, event):
        """Generate and return a random sample value True or False
        given that the parent variables have the values they have in
        event.

        parents is a list of variable names (strings).
        event is a dictionary of variable-name: value pairs.

        >>> cpt = BoolCpt({True: 0.2, False: 0.7})
        >>> cpt.rand(['A'], {'A': True}) in [True, False]
        True
        >>> cpt = BoolCpt({(True, True): 0.1, (True, False): 0.3,
        ...   (False, True): 0.5, (False, False): 0.7})
        >>> cpt.rand(['A', 'B'], {'A': True, 'B': False}) in [True, False]
        True
        """
        return (random() <= self.p(True, parents, event))

def event_values(event, vars):
    """Return a tuple of the values of variables vars in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(vars):
        return event
    return tuple([event[parent] for parent in vars])


#______________________________________________________________________________

def enumerate_joint_ask(X, e, P):
    """Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P. 
    Works for Boolean variables only. [Fig. 13.4].

    X is a string (variable name).
    e is a dictionary of variable-name value pairs.
    P is an instance of JointProbDist."""
    
    Q = ProbDist(X) # probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e] # hidden vars.
    for xi in P.values(X):
        ext = extend(e, X, xi) # copies e and adds X: xi
        Q[xi] = enumerate_joint(Y, ext, P)
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
        update(self, nodes=[], vars=[], evidence={})
        for node in nodes:
            self.add(node)

    def add(self, node):
        self.nodes.append(node)
        self.vars.append(node.variable)

    def observe(self, var, val):
        self.evidence[var] = val

    def variable_node(self, var):
        """Returns the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)

    def variables(self):
        """Returns the list of names of the variables.
        >>> burglary.variables()
        ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']"""
        return [n.variable for n in self.nodes]
    
    def variable_values(self, var):
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
    # be ordered with parents before their children.
    node('Burglary', '', BoolCpt(0.001)),
    node('Earthquake', '', BoolCpt(0.002)),
    node('Alarm', 'Burglary Earthquake',
         BoolCpt({(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})),
    node('JohnCalls', 'Alarm', BoolCpt({T: 0.90, F: 0.05})),
    node('MaryCalls', 'Alarm', BoolCpt({T: 0.70, F: 0.01}))
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
    >>> p.show_approx()
    'False: 0.716, True: 0.284'"""
    Q = ProbDist(X) # empty probability distribution for X
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables(), extend(e, X, xi), bn)
        # Assume that parents precede children in bn.variables.
        # Otherwise, in enumerate_all, the values of Y's parents
        # may be unspecified.
    return Q.normalize()

def enumerate_all (vars, e, bn):
    """Returns the probability that X = xi given e.

    vars is a list of variables, the parents of X in bn.
    e is a dictionary of variable-name: value pairs
    bn is an instance of BayesNet.

    Precondition: no variable in vars precedes its parents."""
    if vars == []:
        return 1.0
    else:
        Y = vars[0]
        rest = vars[1:]

        Ynode = bn.variable_node(Y)
        parents = Ynode.parents
        cpt = Ynode.cpt
        
        if e.has_key(Y):
            y = e[Y]
            cp = cpt.p(y, parents, e) # P(y | parents(Y))
            result = cp * enumerate_all(rest, e, bn)
        else:
            result = 0
            for y in bn.variable_values(Y):
                cp = cpt.p(y, parents, e) # P(y | parents(Y))
                result += cp * enumerate_all(rest, extend(e, Y, y), bn)

        return result

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
    node('Cloudy', '', BoolCpt(0.5)),
    node('Sprinkler', 'Cloudy', BoolCpt({T: 0.10, F: 0.50})),
    node('Rain', 'Cloudy', BoolCpt({T: 0.80, F: 0.20})),
    node('WetGrass', 'Sprinkler Rain',
         BoolCpt({(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00}))])

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
        sample[var] = node.cpt.rand(node.parents, sample)
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
    >>> p.show_approx()
    'False: 0.7, True: 0.3'
    """
    counts = {True: 0, False: 0} # boldface N in Fig. 14.13

    for j in xrange(N):
        sample = prior_sample(bn) # boldface x in Fig. 14.13
        if consistent_with(sample, e):
            counts[sample[X]] += 1
            
    return ProbDist(X, counts)

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

def likelihood_weighting (X, e, bn, N):
    """Returns an estimate of P(X | e).  [Fig. 14.14]

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
    >>> p.show_approx()
    'False: 0.702, True: 0.298'
    """
    weights = {True: 0.0, False: 0.0} # boldface W in Fig. 14.14

    for j in xrange(N):
        sample, weight = weighted_sample(bn, e) # boldface x, w in Fig. 14.14
        sample_X = sample[X] # value of X in sample
        weights[sample_X] += weight

    return ProbDist(X, weights)
    
def weighted_sample (bn, e):
    """Returns an event (a sample) and a weight."""

    event = {} # boldface x in Fig. 14.14
    weight = 1.0 # w in Fig. 14.14

    for node in bn.nodes:
        X = node.variable # X sub i in Fig. 14.14
        parents = node.parents
        cpt = node.cpt
        if e.has_key(X):
            value = e[X]
            event[X] = value
            weight *= cpt.p(value, parents, event)
            
        else:
            event[X] = cpt.rand(parents, event)
        
    return event, weight
    

#_______________________________________________________________________________

# MISSING

# Fig. 14.15: mcmc_ask

__doc__ += """
## We can build up a probability distribution like this (p. 469):
>>> P = ProbDist()
>>> P['sunny'] = 0.7
>>> P['rain'] = 0.2
>>> P['cloudy'] = 0.08
>>> P['snow'] = 0.02

## and query it like this: (Never mind this ELLIPSIS option
##                          added to make the doctest portable.)
>>> P['rain']               #doctest:+ELLIPSIS
0.2...

## A Joint Probability Distribution is dealt with like this (p. 475):
>>> P = JointProbDist(['Toothache', 'Cavity', 'Catch'])
>>> T, F = True, False
>>> P[T, T, T] = 0.108; P[T, T, F] = 0.012; P[F, T, T] = 0.072; P[F, T, F] = 0.008
>>> P[T, F, T] = 0.016; P[T, F, F] = 0.064; P[F, F, T] = 0.144; P[F, F, F] = 0.576

>>> P[T, T, T] 
0.108

## Ask for P(Cavity|Toothache=T)
>>> PC = enumerate_joint_ask('Cavity', {'Toothache': T}, P) 
>>> PC.show_approx()
'False: 0.4, True: 0.6'

>>> 0.6-epsilon < PC[T] < 0.6+epsilon 
True

>>> 0.4-epsilon < PC[F] < 0.4+epsilon 
True
"""
