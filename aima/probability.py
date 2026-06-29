"""Probability models (Chapter 13-15)"""

import copy
import re
from collections import defaultdict
from functools import reduce

from aima.agents import Agent
from aima.utils import *


def DTAgentProgram(belief_state):
    """
    [Figure 13.1]
    A decision-theoretic agent.
    """

    def program(percept):
        belief_state.observe(program.action, percept)
        program.action = max(belief_state.actions(), key=belief_state.expected_outcome_utility)
        return program.action

    program.action = None
    return program


# ______________________________________________________________________________


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, var_name='?', freq=None):
        """If freq is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.var_name = var_name
        self.values = []
        if freq:
            for (v, p) in freq.items():
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
        return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.var_name)


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


# ______________________________________________________________________________


def enumerate_joint_ask(X, e, P):
    """
    [Section 13.3]
    Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P.
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
    return sum([enumerate_joint(rest, extend(e, Y, y), P) for y in P.values(Y)])


# ______________________________________________________________________________
# Independence


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


class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children."""
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)


class DecisionNetwork(BayesNet):
    """An abstract class for a decision network as a wrapper for a BayesNet.
    Represents an agent's current state, its possible actions, reachable states
    and utilities of those states."""

    def __init__(self, action, infer):
        """action: a single action node
        infer: the preferred method to carry out inference on the given BayesNet"""
        super(DecisionNetwork, self).__init__()
        self.action = action
        self.infer = infer

    def best_action(self):
        """Return the best action in the network"""
        return self.action

    def get_utility(self, action, state):
        """Return the utility for a particular action and state in the network"""
        raise NotImplementedError

    def get_expected_utility(self, action, evidence):
        """Compute the expected utility given an action and evidence"""
        u = 0.0
        prob_dist = self.infer(action, evidence, self).prob
        for item, _ in prob_dist.items():
            u += prob_dist[item] * self.get_utility(action, item)

        return u


class InformationGatheringAgent(Agent):
    """
    [Figure 16.9]
    A simple information gathering agent. The agent works by repeatedly selecting
    the observation with the highest information value, until the cost of the next
    observation is greater than its expected benefit."""

    def __init__(self, decnet, infer, initial_evidence=None):
        """decnet: a decision network
        infer: the preferred method to carry out inference on the given decision network
        initial_evidence: initial evidence"""
        self.decnet = decnet
        self.infer = infer
        self.observation = initial_evidence or []
        self.variables = self.decnet.nodes

    def integrate_percept(self, percept):
        """Integrate the given percept into the decision network"""
        raise NotImplementedError

    def execute(self, percept):
        """Execute the information gathering algorithm"""
        self.observation = self.integrate_percept(percept)
        vpis = self.vpi_cost_ratio(self.variables)
        j = max(vpis)
        variable = self.variables[j]

        if self.vpi(variable) > self.cost(variable):
            return self.request(variable)

        return self.decnet.best_action()

    def request(self, variable):
        """Return the value of the given random variable as the next percept"""
        raise NotImplementedError

    def cost(self, var):
        """Return the cost of obtaining evidence through tests, consultants or questions"""
        raise NotImplementedError

    def vpi_cost_ratio(self, variables):
        """Return the VPI to cost ratio for the given variables"""
        v_by_c = []
        for var in variables:
            v_by_c.append(self.vpi(var) / self.cost(var))
        return v_by_c

    def vpi(self, variable):
        """Return VPI for a given variable"""
        vpi = 0.0
        prob_dist = self.infer(variable, self.observation, self.decnet).prob
        for item, _ in prob_dist.items():
            post_prob = prob_dist[item]
            new_observation = list(self.observation)
            new_observation.append(item)
            expected_utility = self.decnet.get_expected_utility(variable, new_observation)
            vpi += post_prob * expected_utility

        vpi -= self.decnet.get_expected_utility(variable, self.observation)
        return vpi


class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string. cpt, the conditional
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
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


class DiscreteBayesNode:
    """A node of a discrete Bayesian network whose variable may take more than two
    values (unlike :class:`BayesNode`, which is boolean).

    ``values`` is the variable's domain. ``cpt`` maps each tuple of parent values
    (ordered as in ``parents``) to the probabilities over ``values`` -- either a
    sequence in domain order or a ``{value: prob}`` dict. A root node uses the
    empty tuple ``()`` as its only key.
    """

    def __init__(self, X, parents, values, cpt):
        if isinstance(parents, str):
            parents = parents.split()
        self.variable = X
        self.parents = parents
        self.values = list(values)
        self.cpt = {}
        for key, dist in cpt.items():
            key = key if isinstance(key, tuple) else (key,)
            assert len(key) == len(self.parents)
            self.cpt[key] = dist if isinstance(dist, dict) else dict(zip(self.values, dist))
        self.children = []

    def p(self, value, event):
        """Return P(X = ``value`` | parents = their values in ``event``)."""
        return self.cpt[tuple(event[parent] for parent in self.parents)][value]

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


class DiscreteBayesNet:
    """A Bayesian network of :class:`DiscreteBayesNode`\\ s (variables with arbitrary
    finite domains). Exact inference works through the generic
    :func:`enumeration_ask` / :func:`elimination_ask`, which rely only on
    ``node.p`` and ``variable_values`` and so work unchanged for multi-valued nodes.
    """

    def __init__(self, node_specs=None):
        self.nodes = []
        self.variables = []
        for spec in node_specs or []:
            self.add(spec)

    def add(self, node_spec):
        """Add a ``DiscreteBayesNode`` (or a ``(name, parents, values, cpt)`` spec);
        its parents must already be in the net and its variable must not."""
        node = node_spec if isinstance(node_spec, DiscreteBayesNode) else DiscreteBayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all(parent in self.variables for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        return self.variable_node(var).values

    def __repr__(self):
        return 'DiscreteBayesNet({0!r})'.format(self.nodes)


def read_bif(source):
    """Parse a Bayesian network in BIF (Bayesian Interchange Format) into a
    :class:`DiscreteBayesNet`. ``source`` is the BIF text or an open file object.

    BIF is the format used by the Bayesian Network Repository
    (https://www.bnlearn.com/bnrepository/), so this lets aima load standard
    multi-valued networks such as the car-insurance ("Insurance") model.
    """
    text = source.read() if hasattr(source, 'read') else source

    # variable NAME { type discrete [ k ] { v1, v2, ... }; }
    domains = {name: [v.strip() for v in vals.split(',') if v.strip()]
               for name, vals in re.findall(r'variable\s+(\w+)\s*\{[^}]*?\{([^}]*)\}', text)}

    # probability ( VAR [ | P1, P2, ... ] ) { table ... ;  |  (pv, ...) p, ... ; }
    specs = {}
    for header, body in re.findall(r'probability\s*\(\s*([^)]*?)\s*\)\s*\{(.*?)\}', text, re.S):
        if '|' in header:
            var, parent_str = header.split('|')
            var, parents = var.strip(), [p.strip() for p in parent_str.split(',') if p.strip()]
        else:
            var, parents = header.strip(), []
        cpt = {}
        table = re.search(r'table\s+([^;]+);', body)
        if table:
            cpt[()] = [float(x) for x in table.group(1).split(',')]
        else:
            for key, probs in re.findall(r'\(([^)]*)\)\s*([^;]+);', body):
                cpt[tuple(v.strip() for v in key.split(',') if v.strip())] = \
                    [float(x) for x in probs.split(',')]
        specs[var] = (parents, domains[var], cpt)

    # add nodes parents-before-children (the BIF order need not be topological)
    net, added = DiscreteBayesNet(), set()

    def add_node(name):
        if name in added:
            return
        parents, values, cpt = specs[name]
        for parent in parents:
            add_node(parent)
        net.add(DiscreteBayesNode(name, parents, values, cpt))
        added.add(name)

    for name in specs:
        add_node(name)
    return net


def insurance():
    """Return the car-insurance ("Insurance") Bayesian network as a
    :class:`DiscreteBayesNet`, loaded from ``aima-data/insurance.bif``.

    This is the 27-variable discrete model of Binder, Koller, Russell & Kanazawa
    (1997) referenced by the AIMA 4e car-insurance case study (Section 16); the
    book notes the discrete conditional distributions are provided in the code
    repository, and this is them.
    """
    return read_bif(open_data('insurance.bif').read())


# Burglary example [Figure 14.2]

T, F = True, False

burglary = BayesNet([('Burglary', '', 0.001),
                     ('Earthquake', '', 0.002),
                     ('Alarm', 'Burglary Earthquake',
                      {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
                     ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
                     ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})])


# ______________________________________________________________________________
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


# ______________________________________________________________________________


def enumeration_ask(X, e, bn):
    """
    [Figure 14.9]
    Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn.
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


def enumerate_all(variables, e, bn):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
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


def elimination_ask(X, e, bn):
    """
    [Figure 14.11]
    Compute bn's P(X|e) by variable elimination.
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
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
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {event_values(e1, variables): node.p(e1[var], e1)
           for e1 in all_events(variables, bn, e)}
    return Factor(variables, cpt)


def pointwise_product(factors, bn):
    """Multiply a sequence of factors together into a single factor over the union of their
    variables, using the Bayes net ``bn`` to enumerate variable values."""
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
        cpt = {event_values(e, variables): self.p(e) * other.p(e) for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {event_values(e, variables): sum(self.p(extend(e, var, val)) for val in bn.variable_values(var))
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """Return my probabilities; must be down to one variable."""
        assert len(self.variables) == 1
        return ProbDist(self.variables[0], {k: v for ((k,), v) in self.cpt.items()})

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

# [Figure 14.12a]: sprinkler network


sprinkler = BayesNet([('Cloudy', '', 0.5),
                      ('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
                      ('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
                      ('WetGrass', 'Sprinkler Rain',
                       {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00})])


# ______________________________________________________________________________


def prior_sample(bn):
    """
    [Figure 14.13]
    Randomly sample from bn's full joint distribution.
    The result is a {variable: value} dict.
    """
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event


# _________________________________________________________________________


def rejection_sampling(X, e, bn, N=10000):
    """
    [Figure 14.14]
    Estimate the probability distribution of variable X given
    evidence e in BayesNet bn, using N samples.
    Raises a ZeroDivisionError if all the N samples are rejected,
    i.e., inconsistent with e.
    >>> random.seed(47)
    >>> rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.7, True: 0.3'
    """
    counts = {x: 0 for x in bn.variable_values(X)}  # bold N in [Figure 14.14]
    for j in range(N):
        sample = prior_sample(bn)  # boldface x in [Figure 14.14]
        if consistent_with(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)


def consistent_with(event, evidence):
    """Is event consistent with the given evidence?"""
    return all(evidence.get(k, v) == v for k, v in event.items())


# _________________________________________________________________________


def likelihood_weighting(X, e, bn, N=10000):
    """
    [Figure 14.15]
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
    event = dict(e)  # boldface x in [Figure 14.15]
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.p(e[Xi], event)
        else:
            event[Xi] = node.sample(event)
    return event, w


# _________________________________________________________________________


def gibbs_ask(X, e, bn, N=1000):
    """[Figure 14.16] Approximate P(X | e) by Gibbs sampling: from a random state
    consistent with the evidence, repeatedly resample each nonevidence variable from
    its Markov blanket and tally how often X takes each value."""
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
    """Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents."""
    Xnode = bn.variable_node(X)
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        ei = extend(e, X, xi)
        # [Equation 14.12]
        Q[xi] = Xnode.p(xi, e) * product(Yj.p(ei[Yj.variable], ei) for Yj in Xnode.children)
    # (assuming a Boolean variable here)
    return probability(Q.normalize()[True])


# _________________________________________________________________________


class HiddenMarkovModel:
    """A Hidden markov model which takes Transition model and Sensor model as inputs"""

    def __init__(self, transition_model, sensor_model, prior=None):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.prior = prior or [0.5, 0.5]

    def sensor_dist(self, ev):
        """Return the sensor (observation) distribution corresponding to the evidence ``ev``:
        the first row of the sensor model when ``ev`` is True, otherwise the second."""
        if ev is True:
            return self.sensor_model[0]
        else:
            return self.sensor_model[1]


def forward(HMM, fv, ev):
    """Perform one forward (filtering) step of an HMM: project the forward message ``fv``
    through the transition model, weight it by the sensor distribution for evidence ``ev``,
    and return the normalized next forward message. [Figure 15.4]"""
    prediction = vector_add(scalar_vector_product(fv[0], HMM.transition_model[0]),
                            scalar_vector_product(fv[1], HMM.transition_model[1]))
    sensor_dist = HMM.sensor_dist(ev)

    return normalize(element_wise_product(sensor_dist, prediction))


def backward(HMM, b, ev):
    """Perform one backward step of an HMM: weight the backward message ``b`` by the sensor
    distribution for evidence ``ev`` and propagate it through the transition model, returning
    the normalized previous backward message. [Figure 15.4]"""
    sensor_dist = HMM.sensor_dist(ev)
    prediction = element_wise_product(sensor_dist, b)

    return normalize(vector_add(scalar_vector_product(prediction[0], HMM.transition_model[0]),
                                scalar_vector_product(prediction[1], HMM.transition_model[1])))


def forward_backward(HMM, ev):
    """
    [Figure 15.4]
    Forward-Backward algorithm for smoothing. Computes posterior probabilities
    of a sequence of states given a sequence of observations.
    """
    t = len(ev)
    ev.insert(0, None)  # to make the code look similar to pseudo code

    fv = [[0.0, 0.0] for _ in range(len(ev))]
    b = [1.0, 1.0]
    sv = [[0, 0] for _ in range(len(ev))]

    fv[0] = HMM.prior

    for i in range(1, t + 1):
        fv[i] = forward(HMM, fv[i - 1], ev[i])
    for i in range(t, -1, -1):
        sv[i - 1] = normalize(element_wise_product(fv[i], b))
        b = backward(HMM, b, ev[i])

    sv = sv[::-1]

    return sv


def viterbi(HMM, ev):
    """
    [Equation 15.11]
    Viterbi algorithm to find the most likely sequence. Computes the best path and the
    corresponding probabilities, given an HMM model and a sequence of observations.
    """
    t = len(ev)
    ev = ev.copy()
    ev.insert(0, None)

    m = [[0.0, 0.0] for _ in range(len(ev) - 1)]

    # the recursion is initialized with m1 = forward(P(X0), e1)
    m[0] = forward(HMM, HMM.prior, ev[1])
    # keep track of maximizing predecessors
    backtracking_graph = []

    for i in range(1, t):
        m[i] = element_wise_product(HMM.sensor_dist(ev[i + 1]),
                                    [max(element_wise_product(HMM.transition_model[0], m[i - 1])),
                                     max(element_wise_product(HMM.transition_model[1], m[i - 1]))])
        backtracking_graph.append([np.argmax(element_wise_product(HMM.transition_model[0], m[i - 1])),
                                   np.argmax(element_wise_product(HMM.transition_model[1], m[i - 1]))])

    # computed probabilities
    ml_probabilities = [0.0] * (len(ev) - 1)
    # most likely sequence
    ml_path = [True] * (len(ev) - 1)

    # the construction of the most likely sequence starts in the final state with the largest probability, and
    # runs backwards; the algorithm needs to store for each xt its predecessor xt-1 maximizing its probability
    i_max = np.argmax(m[-1])

    for i in range(t - 1, -1, -1):
        ml_probabilities[i] = m[i][i_max]
        ml_path[i] = True if i_max == 0 else False
        if i > 0:
            i_max = backtracking_graph[i - 1][i_max]

    return ml_path, ml_probabilities


def baum_welch(HMM, observations, iterations=100):
    """
    [Section 20.3]
    Baum-Welch algorithm: the instance of EM that learns the parameters of a
    Hidden Markov Model (transition model, sensor model and prior) from a single
    sequence of boolean 'observations', starting from the initial guess in 'HMM'.
    Each iteration runs a (scaled) forward-backward pass to compute the smoothed
    state marginals gamma_t(i) = P(X_t=i | e_1:T) and transition marginals
    xi_t(i,j) = P(X_t=i, X_t+1=j | e_1:T) (E-step), then re-estimates every
    parameter as the corresponding normalized expected count (M-step)::

        prior_i     = gamma_0(i)
        A_ij        = sum_t xi_t(i, j) / sum_t gamma_t(i)
        sensor_oi   = sum_{t: e_t = o} gamma_t(i) / sum_t gamma_t(i)

    Returns a new HiddenMarkovModel with the learned parameters.
    """
    A = np.array(HMM.transition_model, dtype=float)
    prior = np.array(HMM.prior, dtype=float)
    # sensor[0] = P(e=True | state), sensor[1] = P(e=False | state)
    sensor = np.array(HMM.sensor_model, dtype=float)
    obs = list(observations)
    n, t_max = len(prior), len(obs)

    for _ in range(iterations):
        # emission vectors b_t(i) = P(e_t | X_t = i), recomputed from current sensor
        B = np.array([sensor[0] if e else sensor[1] for e in obs])

        # E-step: scaled forward (alpha) and backward (beta) messages
        alpha, c = np.zeros((t_max, n)), np.zeros(t_max)
        alpha[0] = prior * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, t_max):
            alpha[t] = B[t] * (alpha[t - 1] @ A)
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.zeros((t_max, n))
        beta[-1] = 1
        for t in range(t_max - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        # smoothed state and transition marginals (normalized, so the per-step
        # scaling factors cancel out)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((t_max - 1, n, n))
        for t in range(t_max - 1):
            xi[t] = alpha[t][:, None] * A * B[t + 1] * beta[t + 1]
            xi[t] /= xi[t].sum()

        # M-step: re-estimate every parameter from the expected counts
        prior = gamma[0]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        mask = np.array(obs, dtype=bool)
        p_true = gamma[mask].sum(axis=0) / gamma.sum(axis=0)
        sensor = np.array([p_true, 1 - p_true])

    return HiddenMarkovModel(A.tolist(), sensor.tolist(), prior.tolist())


# _________________________________________________________________________


def fixed_lag_smoothing(e_t, HMM, d, ev, t):
    """
    [Figure 15.6]
    Smoothing algorithm with a fixed time lag of 'd' steps.
    Computes the smoothed estimate P(X_{t-d} | e_{1:t}) for the slice that lies
    'd' steps in the past, given the evidence sequence ev = [e_1, ..., e_t].
    Returns None when there is not yet enough evidence (t <= d).
    """
    if t <= d:
        return None

    T_model = np.array(HMM.transition_model)

    # forward message advanced over e_1 .. e_{t-d}
    f = HMM.prior
    for i in range(t - d):
        f = forward(HMM, f, ev[i])

    # backward transformation accumulated over the lag window e_{t-d+1} .. e_t
    B = np.eye(len(f))
    for i in range(t - d, t):
        O_i = np.diag(HMM.sensor_dist(ev[i]))
        B = B @ T_model @ O_i

    return normalize((np.array(f) * (B @ np.ones(len(f)))).tolist())


# _________________________________________________________________________


def particle_filtering(e, N, HMM):
    """Particle filtering considering two states variables."""
    dist = [0.5, 0.5]
    # Weight Initialization
    w = [0 for _ in range(N)]
    # STEP 1
    # Propagate one step using transition model given prior state
    dist = vector_add(scalar_vector_product(dist[0], HMM.transition_model[0]),
                      scalar_vector_product(dist[1], HMM.transition_model[1]))
    # Assign state according to probability
    s = ['A' if probability(dist[0]) else 'B' for _ in range(N)]
    w_tot = 0
    # Calculate importance weight given evidence e
    for i in range(N):
        if s[i] == 'A':
            # P(U|A)*P(A)
            w_i = HMM.sensor_dist(e)[0] * dist[0]
        if s[i] == 'B':
            # P(U|B)*P(B)
            w_i = HMM.sensor_dist(e)[1] * dist[1]
        w[i] = w_i
        w_tot += w_i

    # Normalize all the weights
    for i in range(N):
        w[i] = w[i] / w_tot

    # Limit weights to 4 digits
    for i in range(N):
        w[i] = float("{0:.4f}".format(w[i]))

    # STEP 2
    s = weighted_sample_with_replacement(N, s, w)

    return s


# _________________________________________________________________________


class KalmanFilter:
    """
    [Section 15.4]
    Kalman filter for a linear-Gaussian dynamical system. The hidden state
    evolves and is observed according to the linear-Gaussian model

        x_{t+1} = F x_t + noise,   noise ~ N(0, Sigma_x)   (transition model)
        z_t     = H x_t + noise,   noise ~ N(0, Sigma_z)   (sensor model)

    where F is the transition matrix, H the sensor matrix, Sigma_x the
    transition (process) noise covariance and Sigma_z the sensor (measurement)
    noise covariance. Because the family of Gaussians is closed under the
    Bayesian filtering update, the forward message stays Gaussian and is fully
    described by a mean vector and a covariance matrix at every step.
    """

    def __init__(self, transition_model, sensor_model, transition_noise, sensor_noise):
        self.F = np.atleast_2d(transition_model)  # transition matrix
        self.H = np.atleast_2d(sensor_model)  # sensor matrix
        self.Sigma_x = np.atleast_2d(transition_noise)  # transition noise covariance
        self.Sigma_z = np.atleast_2d(sensor_noise)  # sensor noise covariance

    def predict(self, mean, cov):
        """Time update: project the Gaussian estimate one step forward through F."""
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + self.Sigma_x
        return mean, cov

    def update(self, mean, cov, z):
        """Measurement update: condition the predicted Gaussian on observation z."""
        # Kalman gain [Equation 15.21]
        K = cov @ self.H.T @ np.linalg.inv(self.H @ cov @ self.H.T + self.Sigma_z)
        mean = mean + K @ (np.atleast_1d(z) - self.H @ mean)
        cov = (np.eye(cov.shape[0]) - K @ self.H) @ cov
        return mean, cov

    def filter(self, mean, cov, z):
        """One predict-then-update cycle for a single new observation z."""
        mean, cov = self.predict(mean, cov)
        return self.update(mean, cov, z)


def kalman_filter(KF, mean0, cov0, observations):
    """
    [Section 15.4]
    Run the Kalman filter 'KF' over a sequence of 'observations', starting from
    the Gaussian prior N(mean0, cov0). Returns, for each time step, the filtered
    Gaussian estimate as a (mean, covariance) pair.
    """
    mean, cov = np.atleast_1d(mean0).astype(float), np.atleast_2d(cov0).astype(float)
    estimates = []
    for z in observations:
        mean, cov = KF.filter(mean, cov, z)
        estimates.append((mean, cov))

    return estimates


# _________________________________________________________________________


class DynamicBayesNet:
    """
    [Section 15.5]
    A dynamic Bayesian network for a stationary first-order Markov process. It is
    specified by a prior network over the state variables at slice 0 and a single
    transition + sensor network describing, for one time step, the distribution of
    each state variable (given the previous slice) and of each evidence variable
    (given the current slice). The DBN can be 'unrolled' into an ordinary BayesNet
    spanning any number of slices and then queried with the exact inference
    algorithms; in particular filtering is the query for the last state variable
    given the whole evidence sequence.

    Each spec is a (variable, parents, cpt) triple as for a BayesNode. In a
    transition spec, a parent named '<var>_prev' refers to state variable <var> at
    the previous slice; every other parent refers to the current slice.
    """

    def __init__(self, prior, transition, sensors):
        self.prior = prior
        self.transition = transition
        self.sensors = sensors
        self.state_variables = [spec[0] for spec in prior]
        self.evidence_variables = [spec[0] for spec in sensors]

    @staticmethod
    def _rename(parents, t, t_prev):
        """Map the parent names of a slice template to concrete unrolled names."""
        if isinstance(parents, str):
            parents = parents.split()
        return [f'{p[:-len("_prev")]}_{t_prev}' if p.endswith('_prev') else f'{p}_{t}' for p in parents]

    def unroll(self, steps):
        """Unroll the DBN into a BayesNet over slices 0..steps (evidence at 1..steps)."""
        specs = [(f'{var}_0', self._rename(parents, 0, 0), cpt) for var, parents, cpt in self.prior]
        for t in range(1, steps + 1):
            for var, parents, cpt in self.transition + self.sensors:
                specs.append((f'{var}_{t}', self._rename(parents, t, t - 1), cpt))
        return BayesNet(specs)

    def filter(self, evidence, query, infer=elimination_ask):
        """
        Filtering: the posterior over 'query' at the last slice given the whole
        observation sequence. 'evidence' is a list of dicts, one per time step
        t = 1, 2, ..., each mapping evidence variables to their observed values.
        """
        steps = len(evidence)
        net = self.unroll(steps)
        e = {f'{var}_{t}': val for t, obs in enumerate(evidence, 1) for var, val in obs.items()}
        return infer(f'{query}_{steps}', e, net)


# _________________________________________________________________________
# TODO: Implement continuous map for MonteCarlo similar to Fig25.10 from the book


class MCLmap:
    """Map which provides probability distributions and sensor readings.
    Consists of discrete cells which are either an obstacle or empty"""

    def __init__(self, m):
        self.m = m
        self.nrows = len(m)
        self.ncols = len(m[0])
        # list of empty spaces in the map
        self.empty = [(i, j) for i in range(self.nrows) for j in range(self.ncols) if not m[i][j]]

    def sample(self):
        """Returns a random kinematic state possible in the map"""
        pos = random.choice(self.empty)
        # 0N 1E 2S 3W
        orient = random.choice(range(4))
        kin_state = pos + (orient,)
        return kin_state

    def ray_cast(self, sensor_num, kin_state):
        """Returns distance to nearest obstacle or map boundary in the direction of sensor"""
        pos = kin_state[:2]
        orient = kin_state[2]
        # sensor layout when orientation is 0 (towards North)
        #  0
        # 3R1
        #  2
        delta = ((sensor_num % 2 == 0) * (sensor_num - 1), (sensor_num % 2 == 1) * (2 - sensor_num))
        # sensor direction changes based on orientation
        for _ in range(orient):
            delta = (delta[1], -delta[0])
        range_count = 0
        while 0 <= pos[0] < self.nrows and 0 <= pos[1] < self.nrows and not self.m[pos[0]][pos[1]]:
            pos = vector_add(pos, delta)
            range_count += 1
        return range_count


def monte_carlo_localization(a, z, N, P_motion_sample, P_sensor, m, S=None):
    """
    [Figure 25.9]
    Monte Carlo localization algorithm
    """

    def ray_cast(sensor_num, kin_state, m):
        return m.ray_cast(sensor_num, kin_state)

    M = len(z)
    S_ = [0] * N
    W_ = [0] * N
    v = a['v']
    w = a['w']

    if S is None:
        S = [m.sample() for _ in range(N)]

    for i in range(N):
        S_[i] = P_motion_sample(S[i], v, w)
        W_[i] = 1
        for j in range(M):
            z_ = ray_cast(j, S_[i], m)
            W_[i] = W_[i] * P_sensor(z[j], z_)

    S = weighted_sample_with_replacement(N, S_, W_)
    return S
