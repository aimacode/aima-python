"""Probability models (Chapter 12-13)"""

import copy
import random
from collections import defaultdict
from functools import reduce

import numpy as np

from utils4e import product, probability, extend


# ______________________________________________________________________________
# Chapter 12 Qualifying Uncertainty
# 12.1 Acting Under Uncertainty


def DTAgentProgram(belief_state):
    """A decision-theoretic agent. [Figure 12.1]"""

    def program(percept):
        belief_state.observe(program.action, percept)
        program.action = max(belief_state.actions(), key=belief_state.expected_outcome_utility)
        return program.action

    program.action = None
    return program


# ______________________________________________________________________________
# 12.2 Basic Probability Notation


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, varname='?', freqs=None):
        """If freqs is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('{}: ' + numfmt).format(v, p)
                          for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.varname)


# ______________________________________________________________________________
# 12.3 Inference Using Full Joint Distributions


class JointProbDist(ProbDist):
    """A discrete probability distribute over a set of variables.
    >>> P = JointProbDist(['X', 'Y']); P[1, 1] = 0.25
    >>> P[1, 1]
    0.25
    >>> P[dict(X=0, Y=1)] = 0.5
    >>> P[dict(X=0, Y=1)]
    0.5"""

    def __init__(self, variables):
        self.prob = {}
        self.variables = variables
        self.vals = defaultdict(list)

    def __getitem__(self, values):
        """Given a tuple or dict of values, return P(values)."""
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

    def __setitem__(self, values, p):
        """Set P(values) = p. Values can be a tuple or a dict; it must
        have a value for each of the variables in the joint. Also keep track
        of the values we have seen so far for each variable."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        """Return the set of possible values for a variable."""
        return self.vals[var]

    def __repr__(self):
        return "P({})".format(self.variables)


def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])


def enumerate_joint_ask(X, e, P):
    """Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P. [Section 12.3]
    >>> P = JointProbDist(['X', 'Y'])
    >>> P[0,0] = 0.25; P[0,1] = 0.5; P[1,1] = P[2,1] = 0.125
    >>> enumerate_joint_ask('X', dict(Y=1), P).show_approx()
    '0: 0.667, 1: 0.167, 2: 0.167'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)  # probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e]  # hidden variables.
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()


def enumerate_joint(variables, e, P):
    """Return the sum of those entries in P consistent with e,
    provided variables is P's remaining variables (the ones not in e)."""
    if not variables:
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint(rest, extend(e, Y, y), P)
                for y in P.values(Y)])


# ______________________________________________________________________________
# 12.4 Independence


def is_independent(variables, P):
    """
    Return whether a list of variables are independent given their distribution P
    P is an instance of JoinProbDist
    >>> P = JointProbDist(['X', 'Y'])
    >>> P[0,0] = 0.25; P[0,1] = 0.5; P[1,1] = P[1,0] = 0.125
    >>> is_independent(['X', 'Y'], P)
    False
    """
    for var in variables:
        event_vars = variables[:]
        event_vars.remove(var)
        event = {}
        distribution = enumerate_joint_ask(var, event, P)
        events = gen_possible_events(event_vars, P)
        for e in events:
            conditional_distr = enumerate_joint_ask(var, e, P)
            if conditional_distr.prob != distribution.prob:
                return False
    return True


def gen_possible_events(vars, P):
    """Generate all possible events of a collection of vars according to distribution of P"""
    events = []

    def backtrack(vars, P, temp):
        if not vars:
            events.append(temp)
            return
        var = vars[0]
        for val in P.values(var):
            temp[var] = val
            backtrack([v for v in vars if v != var], P, copy.copy(temp))

    backtrack(vars, P, {})
    return events


# ______________________________________________________________________________
# Chapter 13 Probabilistic Reasoning
# 13.1 Representing Knowledge in an Uncertain Domain


class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=None):
        """
        Nodes must be ordered with parents before children.
        :param node_specs: an nested iterable object, each element contains (variable name, parents name, cpt)
                           for each node
        """

        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """
        Add a node to the net. Its parents must already be in the
        net, and its variable must not.
        Initialize Bayes nodes by detecting the length of input node specs
        """
        if len(node_spec) >= 5:
            node = ContinuousBayesNode(*node_spec)
        else:
            node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """
        Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'
        """
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)


class BayesNode:
    """
    A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet.
    """

    def __init__(self, X, parents, cpt):
        """
        :param X: variable name,
        :param parents: a sequence of variable names or a space-separated string. Representing the names of parent nodes
        :param cpt: the conditional probability table, takes one of these forms:

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
        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """
        Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375
        """
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """
        Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents.
        """
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


# Burglary example [Figure 13 .2]


T, F = True, False

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake',
     {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
])


# ______________________________________________________________________________
# Section 13.2. The Semantics of Bayesian Networks
# Bayesian nets with continuous variables


def gaussian_probability(param, event, value):
    """
    Gaussian probability of a continuous Bayesian network node on condition of
    certain event and the parameters determined by the event
    :param param: parameters determined by discrete parent events of current node
    :param event: a dict, continuous event of current node, the values are used
                  as parameters in calculating distribution
    :param value: float, the value of current continuous node
    :return: float, the calculated probability
    >>> param = {'sigma':0.5, 'b':1, 'a':{'h1':0.5, 'h2': 1.5}}
    >>> event = {'h1':0.6, 'h2': 0.3}
    >>> gaussian_probability(param, event, 1)
    0.2590351913317835
    """

    assert isinstance(event, dict)
    assert isinstance(param, dict)
    buff = 0
    for k, v in event.items():
        # buffer varianle to calculate h1*a_h1 + h2*a_h2
        buff += param['a'][k] * v
    res = 1 / (param['sigma'] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((value - buff - param['b']) / param['sigma']) ** 2)
    return res


def logistic_probability(param, event, value):
    """
    Logistic probability of a discrete node in Bayesian network with continuous parents,
    :param param: a dict, parameters determined by discrete parents of current node
    :param event: a dict, names and values of continuous parent variables of current node
    :param value: boolean, True or False
    :return: int, probability
    """

    buff = 1
    for _, v in event.items():
        # buffer variable to calculate (value-mu)/sigma

        buff *= (v - param['mu']) / param['sigma']
    p = 1 - 1 / (1 + np.exp(-4 / np.sqrt(2 * np.pi) * buff))
    return p if value else 1 - p


class ContinuousBayesNode:
    """ A Bayesian network node with continuous distribution or with continuous distributed parents """

    def __init__(self, name, d_parents, c_parents, parameters, type):
        """
        A continuous Bayesian node has two types of parents: discrete and continuous.
        :param d_parents: str, name of discrete parents, value of which determines distribution parameters
        :param c_parents: str, name of continuous parents, value of which is used to calculate distribution
        :param parameters: a dict, parameters for distribution of current node, keys corresponds to discrete parents
        :param type: str, type of current node's value, either 'd' (discrete) or 'c'(continuous)
        """

        self.parameters = parameters
        self.type = type
        self.d_parents = d_parents.split()
        self.c_parents = c_parents.split()
        self.parents = self.d_parents + self.c_parents
        self.variable = name
        self.children = []

    def continuous_p(self, value, c_event, d_event):
        """
        Probability given the value of current node and its parents
        :param c_event: event of continuous nodes
        :param d_event: event of discrete nodes
        """
        assert isinstance(c_event, dict)
        assert isinstance(d_event, dict)

        d_event_vals = event_values(d_event, self.d_parents)
        if len(d_event_vals) == 1:
            d_event_vals = d_event_vals[0]
        param = self.parameters[d_event_vals]
        if self.type == "c":
            p = gaussian_probability(param, c_event, value)
        if self.type == "d":
            p = logistic_probability(param, c_event, value)
        return p


# harvest-buy example. Figure 13.5


harvest_buy = BayesNet([
    ('Subsidy', '', 0.001),
    ('Harvest', '', 0.002),
    ('Cost', 'Subsidy', 'Harvest',
     {True: {'sigma': 0.5, 'b': 1, 'a': {'Harvest': 0.5}},
      False: {'sigma': 0.6, 'b': 1, 'a': {'Harvest': 0.5}}}, 'c'),
    ('Buys', '', 'Cost', {T: {'mu': 0.5, 'sigma': 0.5}, F: {'mu': 0.6, 'sigma': 0.6}}, 'd')])


# ______________________________________________________________________________
# 13.3 Exact Inference in Bayesian Networks
# 13.3.1 Inference by enumeration


def enumeration_ask(X, e, bn):
    """
    Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn. [Figure 13.10]
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'
    """

    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


def enumerate_all(variables, e, bn):
    """
    Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables.
    """

    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))


# ______________________________________________________________________________
# 13.3.2 The variable elimination algorithm


def elimination_ask(X, e, bn):
    """
    Compute bn's P(X|e) by variable elimination. [Figure 13.12]
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()


def is_hidden(var, X, e):
    """Is var a hidden variable when querying P(X|e)?"""
    return var != X and var not in e


def make_factor(var, e, bn):
    """
    Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables.
    """
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {event_values(e1, variables): node.p(e1[var], e1)
           for e1 in all_events(variables, bn, e)}
    return Factor(variables, cpt)


def pointwise_product(factors, bn):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def sum_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.variables else result).append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn):
        """Multiply two factors, combining their variables."""
        variables = list(set(self.variables) | set(other.variables))
        cpt = {event_values(e, variables): self.p(e) * other.p(e)
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {event_values(e, variables): sum(self.p(extend(e, var, val))
                                               for val in bn.variable_values(var))
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """Return my probabilities; must be down to one variable."""
        assert len(self.variables) == 1
        return ProbDist(self.variables[0],
                        {k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        """Look up my value tabulated for e."""
        return self.cpt[event_values(e, self.variables)]


def all_events(variables, bn, e):
    """Yield every way of extending e with values for all variables."""
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)


# ______________________________________________________________________________
# 13.3.4 Clustering algorithms
# [Figure 13.14a]: sprinkler network


sprinkler = BayesNet([
    ('Cloudy', '', 0.5),
    ('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
    ('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
    ('WetGrass', 'Sprinkler Rain',
     {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00})])


# ______________________________________________________________________________
# 13.4 Approximate Inference for Bayesian Networks
# 13.4.1 Direct sampling methods


def prior_sample(bn):
    """
    Randomly sample from bn's full joint distribution. The result
    is a {variable: value} dict. [Figure 13.15]
    """
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event


# _________________________________________________________________________


def rejection_sampling(X, e, bn, N=10000):
    """
    [Figure 13.16]
    Estimate the probability distribution of variable X given
    evidence e in BayesNet bn, using N samples.
    Raises a ZeroDivisionError if all the N samples are rejected,
    i.e., inconsistent with e.
    >>> random.seed(47)
    >>> rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.7, True: 0.3'
    """
    counts = {x: 0 for x in bn.variable_values(X)}  # bold N in [Figure 13.16]
    for j in range(N):
        sample = prior_sample(bn)  # boldface x in [Figure 13.16]
        if consistent_with(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)


def consistent_with(event, evidence):
    """Is event consistent with the given evidence?"""
    return all(evidence.get(k, v) == v
               for k, v in event.items())


# _________________________________________________________________________


def likelihood_weighting(X, e, bn, N=10000):
    """
    [Figure 13.17]
    Estimate the probability distribution of variable X given
    evidence e in BayesNet bn.
    >>> random.seed(1017)
    >>> likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.702, True: 0.298'
    """

    W = {x: 0 for x in bn.variable_values(X)}
    for j in range(N):
        sample, weight = weighted_sample(bn, e)  # boldface x, w in [Figure 14.15]
        W[sample[X]] += weight
    return ProbDist(X, W)


def weighted_sample(bn, e):
    """
    Sample an event from bn that's consistent with the evidence e;
    return the event and its weight, the likelihood that the event
    accords to the evidence.
    """

    w = 1
    event = dict(e)  # boldface x in [Figure 13.17]
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.p(e[Xi], event)
        else:
            event[Xi] = node.sample(event)
    return event, w


# _________________________________________________________________________
# 13.4.2 Inference by Markov chain simulation


def gibbs_ask(X, e, bn, N=1000):
    """[Figure 13.19]"""
    assert X not in e, "Query variable must be distinct from evidence"
    counts = {x: 0 for x in bn.variable_values(X)}  # bold N in [Figure 14.16]
    Z = [var for var in bn.variables if var not in e]
    state = dict(e)  # boldface x in [Figure 14.16]
    for Zi in Z:
        state[Zi] = random.choice(bn.variable_values(Zi))
    for j in range(N):
        for Zi in Z:
            state[Zi] = markov_blanket_sample(Zi, state, bn)
            counts[state[X]] += 1
    return ProbDist(X, counts)


def markov_blanket_sample(X, e, bn):
    """
    Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents.
    """
    Xnode = bn.variable_node(X)
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        ei = extend(e, X, xi)
        # [Equation 13.12:]
        Q[xi] = Xnode.p(xi, e) * product(Yj.p(ei[Yj.variable], ei)
                                         for Yj in Xnode.children)
    # (assuming a Boolean variable here)
    return probability(Q.normalize()[True])


# _________________________________________________________________________
# 13.4.3 Compiling approximate inference


class complied_burglary:
    """compiled version of burglary network"""

    def Burglary(self, sample):
        if sample['Alarm']:
            if sample['Earthquake']:
                return probability(0.00327)
            else:
                return probability(0.485)
        else:
            if sample['Earthquake']:
                return probability(7.05e-05)
            else:
                return probability(6.01e-05)

    def Earthquake(self, sample):
        if sample['Alarm']:
            if sample['Burglary']:
                return probability(0.0020212)
            else:
                return probability(0.36755)
        else:
            if sample['Burglary']:
                return probability(0.0016672)
            else:
                return probability(0.0014222)

    def MaryCalls(self, sample):
        if sample['Alarm']:
            return probability(0.7)
        else:
            return probability(0.01)

    def JongCalls(self, sample):
        if sample['Alarm']:
            return probability(0.9)
        else:
            return probability(0.05)

    def Alarm(self, sample):
        raise NotImplementedError
