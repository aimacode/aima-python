"""Probability models. (Chapter 13-15)
"""

from utils import *
from logic import extend
from random import choice, seed

#______________________________________________________________________________

def DTAgentProgram(belief_state):
    "A decision-theoretic agent. [Fig. 13.1]"
    def program(percept):
        belief_state.observe(program.action, percept)
        program.action = argmax(belief_state.actions(),
                                belief_state.expected_outcome_utility)
        return program.action
    program.action = None
    return program

#______________________________________________________________________________

class ProbDist:
    """A discrete probability distribution.  You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
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
        try: return self.prob[val]
        except KeyError: return 0

    def __setitem__(self, val, p):
        "Set P(val) = p."
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

    def show_approx(self, numfmt='%.3g'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('%s: ' + numfmt) % (v, p)
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
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

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

def event_values(event, vars):
    """Return a tuple of the values of variables vars in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(vars):
        return event
    else:
        return tuple([event[var] for var in vars])

#______________________________________________________________________________

def enumerate_joint_ask(X, e, P):
    """Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P. [Section 13.3]
    >>> P = JointProbDist(['X', 'Y'])
    >>> P[0,0] = 0.25; P[0,1] = 0.5; P[1,1] = P[2,1] = 0.125
    >>> enumerate_joint_ask('X', dict(Y=1), P).show_approx()
    '0: 0.667, 1: 0.167, 2: 0.167'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X) # probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e] # hidden vars.
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()

def enumerate_joint(vars, e, P):
    """Return the sum of those entries in P consistent with e,
    provided vars is P's remaining variables (the ones not in e)."""
    if not vars:
        return P[e]
    Y, rest = vars[0], vars[1:]
    return sum([enumerate_joint(rest, extend(e, Y, y), P)
                for y in P.values(Y)])

#______________________________________________________________________________

class BayesNet:
    "Bayesian network containing only boolean-variable nodes."

    def __init__(self, node_specs=[]):
        "nodes must be ordered with parents before children."
        update(self, nodes=[], vars=[])
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.vars
        assert every(lambda parent: parent in self.vars, node.parents)
        self.nodes.append(node)
        self.vars.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)

    def variable_values(self, var):
        "Return the domain of var."
        return [True, False]

    def __repr__(self):
        return 'BayesNet(%r)' % self.nodes

class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string.  cpt, the conditional
        probability table, takes one of these forms:

        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.

        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.

        * A dict {(v1, v2, ...): p, ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = p. Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.

        In all cases the probability of X being false is left implicit,
        since it follows from P(X=true).

        >>> X = BayesNode('X', '', 0.2)
        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str): parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)): # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            if cpt and isinstance(cpt.keys()[0], bool): # one parent, 1-tuple
                cpt = dict(((v,), p) for v, p in cpt.items())

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert every(lambda v: isinstance(v, bool), vs)
            assert 0 <= p <= 1

        update(self, variable=X, parents=parents, cpt=cpt, children=[])

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return if_(value, ptrue, 1 - ptrue)

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_vars. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))

# Burglary example [Fig. 14.2]

T, F = True, False

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake',
         {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])

#______________________________________________________________________________

def enumeration_ask(X, e, bn):
    """Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn. [Fig. 14.9]
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.vars, extend(e, X, xi), bn)
    return Q.normalize()

def enumerate_all(vars, e, bn):
    """Return the sum of those entries in P(vars | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than vars). Parents must precede children in vars."""
    if not vars:
        return 1.0
    Y, rest = vars[0], vars[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))

#______________________________________________________________________________

def elimination_ask(X, e, bn):
    """Compute bn's P(X|e) by variable elimination. [Fig. 14.11]
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.vars):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()

def is_hidden(var, X, e):
    "Is var a hidden variable when querying P(X|e)?"
    return var != X and var not in e

def make_factor(var, e, bn):
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    node = bn.variable_node(var)
    vars = [X for X in [var] + node.parents if X not in e]
    cpt = dict((event_values(e1, vars), node.p(e1[var], e1))
               for e1 in all_events(vars, bn, e))
    return Factor(vars, cpt)

def pointwise_product(factors, bn):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)

def sum_out(var, factors, bn):
    "Eliminate var from all factors by summing over its values."
    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.vars else result).append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result

class Factor:
    "A factor in a joint distribution."

    def __init__(self, vars, cpt):
        update(self, vars=vars, cpt=cpt)

    def pointwise_product(self, other, bn):
        "Multiply two factors, combining their variables."
        vars = list(set(self.vars) | set(other.vars))
        cpt = dict((event_values(e, vars), self.p(e) * other.p(e))
                   for e in all_events(vars, bn, {}))
        return Factor(vars, cpt)

    def sum_out(self, var, bn):
        "Make a factor eliminating var by summing over its values."
        vars = [X for X in self.vars if X != var]
        cpt = dict((event_values(e, vars),
                    sum(self.p(extend(e, var, val))
                        for val in bn.variable_values(var)))
                   for e in all_events(vars, bn, {}))
        return Factor(vars, cpt)

    def normalize(self):
        "Return my probabilities; must be down to one variable."
        assert len(self.vars) == 1
        return ProbDist(self.vars[0],
                        dict((k, v) for ((k,), v) in self.cpt.items()))

    def p(self, e):
        "Look up my value tabulated for e."
        return self.cpt[event_values(e, self.vars)]

def all_events(vars, bn, e):
    "Yield every way of extending e with values for all vars."
    if not vars:
        yield e
    else:
        X, rest = vars[0], vars[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)

#______________________________________________________________________________

# Fig. 14.12a: sprinkler network

sprinkler = BayesNet([
    ('Cloudy', '', 0.5),
    ('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
    ('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
    ('WetGrass', 'Sprinkler Rain',
         {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00})])

#______________________________________________________________________________

def prior_sample(bn):
    """Randomly sample from bn's full joint distribution. The result
    is a {variable: value} dict. [Fig. 14.13]"""
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event

#_______________________________________________________________________________

def rejection_sampling(X, e, bn, N):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn, using N samples.  [Fig. 14.14]
    Raises a ZeroDivisionError if all the N samples are rejected,
    i.e., inconsistent with e.
    >>> seed(47)
    >>> rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.7, True: 0.3'
    """
    counts = dict((x, 0) for x in bn.variable_values(X)) # bold N in Fig. 14.14
    for j in xrange(N):
        sample = prior_sample(bn) # boldface x in Fig. 14.14
        if consistent_with(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)

def consistent_with(event, evidence):
    "Is event consistent with the given evidence?"
    return every(lambda (k, v): evidence.get(k, v) == v,
                 event.items())

#_______________________________________________________________________________

def likelihood_weighting(X, e, bn, N):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn.  [Fig. 14.15]
    >>> seed(1017)
    >>> likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.702, True: 0.298'
    """
    W = dict((x, 0) for x in bn.variable_values(X))
    for j in xrange(N):
        sample, weight = weighted_sample(bn, e) # boldface x, w in Fig. 14.15
        W[sample[X]] += weight
    return ProbDist(X, W)

def weighted_sample(bn, e):
    """Sample an event from bn that's consistent with the evidence e;
    return the event and its weight, the likelihood that the event
    accords to the evidence."""
    w = 1
    event = dict(e) # boldface x in Fig. 14.15
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.p(e[Xi], event)
        else:
            event[Xi] = node.sample(event)
    return event, w

#_______________________________________________________________________________

def gibbs_ask(X, e, bn, N):
    """[Fig. 14.16]
    >>> seed(1017)
    >>> gibbs_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary, 1000
    ...  ).show_approx()
    'False: 0.738, True: 0.262'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    counts = dict((x, 0) for x in bn.variable_values(X)) # bold N in Fig. 14.16
    Z = [var for var in bn.vars if var not in e]
    state = dict(e) # boldface x in Fig. 14.16
    for Zi in Z:
        state[Zi] = choice(bn.variable_values(Zi))
    for j in xrange(N):
        for Zi in Z:
            state[Zi] = markov_blanket_sample(Zi, state, bn)
            counts[state[X]] += 1
    return ProbDist(X, counts)

def markov_blanket_sample(X, e, bn):
    """Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents."""
    Xnode = bn.variable_node(X)
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        ei = extend(e, X, xi)
        # [Equation 14.12:]
        Q[xi] = Xnode.p(xi, e) * product(Yj.p(ei[Yj.variable], ei)
                                         for Yj in Xnode.children)
    return probability(Q.normalize()[True]) # (assuming a Boolean variable here)

#_______________________________________________________________________________

def forward_backward(ev, prior):
    """[Fig. 15.4]"""
    unimplemented()

def fixed_lag_smoothing(e_t, hmm, d):
    """[Fig. 15.6]"""
    unimplemented()

def particle_filtering(e, N, dbn):
    """[Fig. 15.17]"""
    unimplemented()

#_______________________________________________________________________________
__doc__ += """
# We can build up a probability distribution like this (p. 469):
>>> P = ProbDist()
>>> P['sunny'] = 0.7
>>> P['rain'] = 0.2
>>> P['cloudy'] = 0.08
>>> P['snow'] = 0.02

# and query it like this:  (Never mind this ELLIPSIS option
#                           added to make the doctest portable.)
>>> P['rain']               #doctest:+ELLIPSIS
0.2...

# A Joint Probability Distribution is dealt with like this (Fig. 13.3):
>>> P = JointProbDist(['Toothache', 'Cavity', 'Catch'])
>>> T, F = True, False
>>> P[T, T, T] = 0.108; P[T, T, F] = 0.012; P[F, T, T] = 0.072; P[F, T, F] = 0.008
>>> P[T, F, T] = 0.016; P[T, F, F] = 0.064; P[F, F, T] = 0.144; P[F, F, F] = 0.576

>>> P[T, T, T]
0.108

# Ask for P(Cavity|Toothache=T)
>>> PC = enumerate_joint_ask('Cavity', {'Toothache': T}, P)
>>> PC.show_approx()
'False: 0.4, True: 0.6'

>>> 0.6-epsilon < PC[T] < 0.6+epsilon
True

>>> 0.4-epsilon < PC[F] < 0.4+epsilon
True
"""
