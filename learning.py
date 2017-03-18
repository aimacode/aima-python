"""Learn to estimate functions from examples. (Chapters 18-20)"""

from utils import (
    removeall, unique, product, mode, argmax, argmax_random_tie, isclose,
    dotproduct, vector_add, scalar_vector_product, weighted_sample_with_replacement,
    weighted_sampler, num_or_str, normalize, clip, sigmoid, print_table, DataFile
)

import copy
import heapq
import math
import random

from statistics import mean
from collections import defaultdict, Counter

# ______________________________________________________________________________

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))


def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])


def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])


def manhattan_distance(predictions, targets):
    return sum([abs(p - t) for p, t in zip(predictions, targets)])


def mean_boolean_error(predictions, targets):
    return mean(int(p != t) for p, t in zip(predictions, targets))

def hamming_distance(predictions, targets):
    return sum(p != t for p, t in zip(predictions, targets))

# ______________________________________________________________________________


class DataSet:
    """A data set for a machine learning problem.  It has the following fields:

    d.examples   A list of examples.  Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attrnames  Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.setproblem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a nonnegative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, examples=None, attrs=None, attrnames=None, target=-1,
                 inputs=None, values=None, distance=mean_boolean_error,
                 name='', source='', exclude=()):
        """Accepts any of DataSet's fields.  Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .setproblem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        if values is None:
            self.got_values_flag = False
        else:
            self.got_values_flag = True

        # Initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(DataFile(name + '.csv').read())
        else:
            self.examples = examples
        # Attrs are the indices of examples, unless otherwise stated.
        if attrs is None and self.examples is not None:
            attrs = list(range(len(self.examples[0])))
        self.attrs = attrs
        # Initialize .attrnames from string, list, or by default
        if isinstance(attrnames, str):
            self.attrnames = attrnames.split()
        else:
            self.attrnames = attrnames or attrs
        self.setproblem(target, inputs=inputs, exclude=exclude)

    def setproblem(self, target, inputs=None, exclude=()):
        """Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""
        self.target = self.attrnum(target)
        exclude = map(self.attrnum, exclude)
        if inputs:
            self.inputs = removeall(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs
                           if a != self.target and a not in exclude]
        if not self.values:
            self.values = list(map(unique, zip(*self.examples)))
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attrnames) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attrnames[a], example))

    def attrnum(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attrnames.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None
                for i, attr_i in enumerate(example)]

    def classes_to_numbers(self,classes=None):
        """Converts class names to numbers."""
        if not classes:
            # If classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])
            
    def remove_examples(self,value=""):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(
            self.name, len(self.examples), len(self.attrs))

# ______________________________________________________________________________


def parse_csv(input, delim=','):
    r"""Input is a string consisting of lines, each line has comma-delimited
    fields.  Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]"""
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]

# ______________________________________________________________________________


class CountingProbDist:
    """A probability distribution formed by observing and counting examples.
    If p is an instance of this class and o is an observed value, then
    there are 3 main operations:
    p.add(o) increments the count for observation o by 1.
    p.sample() returns a random element from the distribution.
    p[o] returns the probability for o (as in a regular ProbDist)."""

    def __init__(self, observations=[], default=0):
        """Create a distribution, and optionally add in some observations.
        By default this is an unsmoothed distribution, but saying default=1,
        for example, gives you add-one smoothing."""
        self.dictionary = {}
        self.n_obs = 0.0
        self.default = default
        self.sampler = None

        for o in observations:
            self.add(o)

    def add(self, o):
        """Add an observation o to the distribution."""
        self.smooth_for(o)
        self.dictionary[o] += 1
        self.n_obs += 1
        self.sampler = None

    def smooth_for(self, o):
        """Include o among the possible observations, whether or not
        it's been observed yet."""
        if o not in self.dictionary:
            self.dictionary[o] = self.default
            self.n_obs += self.default
            self.sampler = None

    def __getitem__(self, item):
        """Return an estimate of the probability of item."""
        self.smooth_for(item)
        return self.dictionary[item] / self.n_obs

    # (top() and sample() are not used in this module, but elsewhere.)

    def top(self, n):
        """Return (count, obs) tuples for the n most frequent observations."""
        return heapq.nlargest(n, [(v, k) for (k, v) in self.dictionary.items()])

    def sample(self):
        """Return a random sample from the distribution."""
        if self.sampler is None:
            self.sampler = weighted_sampler(list(self.dictionary.keys()),
                                            list(self.dictionary.values()))
        return self.sampler()

# ______________________________________________________________________________


def PluralityLearner(dataset):
    """A very dumb algorithm: always pick the result that was most popular
    in the training data.  Makes a baseline for comparison."""
    most_popular = mode([e[dataset.target] for e in dataset.examples])

    def predict(example):
        """Always return same result: the most popular from the training set."""
        return most_popular
    return predict

# ______________________________________________________________________________


def NaiveBayesLearner(dataset):
    """Just count how many times each value of each input attribute
    occurs, conditional on the target value. Count the different
    target values too."""

    targetvals = dataset.values[dataset.target]
    target_dist = CountingProbDist(targetvals)
    attr_dists = {(gv, attr): CountingProbDist(dataset.values[attr])
                  for gv in targetvals
                  for attr in dataset.inputs}
    for example in dataset.examples:
        targetval = example[dataset.target]
        target_dist.add(targetval)
        for attr in dataset.inputs:
            attr_dists[targetval, attr].add(example[attr])

    def predict(example):
        """Predict the target value for example. Consider each possible value,
        and pick the most likely by looking at each attribute independently."""
        def class_probability(targetval):
            return (target_dist[targetval] *
                    product(attr_dists[targetval, attr][example[attr]]
                            for attr in dataset.inputs))
        return argmax(targetvals, key=class_probability)

    return predict

# ______________________________________________________________________________


def NearestNeighborLearner(dataset, k=1):
    """k-NearestNeighbor: the k nearest neighbors vote."""
    def predict(example):
        """Find the k closest items, and have them vote for the best."""
        best = heapq.nsmallest(k, ((dataset.distance(e, example), e)
                                   for e in dataset.examples))
        return mode(e[dataset.target] for (d, e) in best)
    return predict

# ______________________________________________________________________________


class DecisionFork:
    """A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values."""

    def __init__(self, attr, attrname=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attrname = attrname or attr
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attrvalue = example[self.attr]
        return self.branches[attrvalue](example)

    def add(self, val, subtree):
        """Add a branch.  If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attrname
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return ('DecisionFork({0!r}, {1!r}, {2!r})'
                .format(self.attr, self.attrname, self.branches))


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self, indent=0):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)

# ______________________________________________________________________________


def DecisionTreeLearner(dataset):
    """[Figure 18.5]"""

    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = DecisionFork(A, dataset.attrnames[A])
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def plurality_value(examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        popular = argmax_random_tie(values[target],
                                    key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have attr = val."""
        return sum(e[attr] == val for e in examples) #count(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs,
                                 key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""
        def I(examples):
            return information_content([count(target, v, examples)
                                        for v in values[target]])
        N = float(len(examples))
        remainder = sum((len(examples_i) / N) * I(examples_i)
                        for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v])
                for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(removeall(0, values))
    return sum(-p * math.log2(p) for p in probabilities)

# ______________________________________________________________________________

# A decision list is implemented as a list of (test, value) pairs.


def DecisionListLearner(dataset):
    """[Figure 18.11]"""

    def decision_list_learning(examples):
        if not examples:
            return [(True, False)]
        t, o, examples_t = find_examples(examples)
        if not t:
            raise Exception
        return [(t, o)] + decision_list_learning(examples - examples_t)

    def find_examples(examples):
        """Find a set of examples that all have the same outcome under
        some test. Return a tuple of the test, outcome, and examples."""
        raise NotImplementedError

    def passes(example, test):
        """Does the example pass the test?"""
        raise NotImplementedError

    def predict(example):
        """Predict the outcome for the first passing test."""
        for test, outcome in predict.decision_list:
            if passes(example, test):
                return outcome
    predict.decision_list = decision_list_learning(set(dataset.examples))

    return predict

# ______________________________________________________________________________


def NeuralNetLearner(dataset, hidden_layer_sizes=[3],
                     learning_rate=0.01, epochs=100):
    """Layered feed-forward network.
    hidden_layer_sizes: List of number of hidden units per hidden layer
    learning_rate: Learning rate of gradient descent
    epochs: Number of passes over the dataset
    """

    i_units = len(dataset.inputs)
    o_units = 1  # As of now, dataset.target gives only one index.

    # construct a network
    raw_net = network(i_units, hidden_layer_sizes, o_units)
    learned_net = BackPropagationLearner(dataset, raw_net,
                                         learning_rate, epochs)

    def predict(example):

        # Input nodes
        i_nodes = learned_net[0]

        # Activate input layer
        for v, n in zip(example, i_nodes):
            n.value = v

        # Forward pass
        for layer in learned_net[1:]:
            for node in layer:
                inc = [n.value for n in node.inputs]
                in_val = dotproduct(inc, node.weights)
                node.value = node.activation(in_val)

        # Hypothesis
        o_nodes = learned_net[-1]
        pred = [o_nodes[i].value for i in range(o_units)]
        return 1 if pred[0] >= 0.5 else 0

    return predict


class NNUnit:
    """Single Unit of Multiple Layer Neural Network
    inputs: Incoming connections
    weights: Weights to incoming connections
    """

    def __init__(self, weights=None, inputs=None):
        self.weights = []
        self.inputs = []
        self.value = None
        self.activation = sigmoid


def network(input_units, hidden_layer_sizes, output_units):
    """Create Directed Acyclic Network of given number layers.
    hidden_layers_sizes : List number of neuron units in each hidden layer
    excluding input and output layers
    """
    # Check for PerceptronLearner
    if hidden_layer_sizes:
        layers_sizes = [input_units] + hidden_layer_sizes + [output_units]
    else:
        layers_sizes = [input_units] + [output_units]

    net = [[NNUnit() for n in range(size)]
           for size in layers_sizes]
    n_layers = len(net)

    # Make Connection
    for i in range(1, n_layers):
        for n in net[i]:
            for k in net[i-1]:
                n.inputs.append(k)
                n.weights.append(0)
    return net


def BackPropagationLearner(dataset, net, learning_rate, epochs):
    """[Figure 18.23] The back-propagation algorithm for multilayer network"""
    # Initialise weights
    for layer in net:
        for node in layer:
            node.weights = [random.uniform(-0.5, 0.5)
                            for i in range(len(node.weights))]

    examples = dataset.examples
    '''
    As of now dataset.target gives an int instead of list,
    Changing dataset class will have effect on all the learners.
    Will be taken care of later
    '''
    idx_t = [dataset.target]
    idx_i = dataset.inputs
    n_layers = len(net)
    o_nodes = net[-1]
    i_nodes = net[0]

    for epoch in range(epochs):
        # Iterate over each example
        for e in examples:
            i_val = [e[i] for i in idx_i]
            t_val = [e[i] for i in idx_t]
            # Activate input layer
            for v, n in zip(i_val, i_nodes):
                n.value = v

            # Forward pass
            for layer in net[1:]:
                for node in layer:
                    inc = [n.value for n in node.inputs]
                    in_val = dotproduct(inc, node.weights)
                    node.value = node.activation(in_val)

            # Initialize delta
            delta = [[] for i in range(n_layers)]

            # Compute outer layer delta
            o_units = len(o_nodes)
            err = [t_val[i] - o_nodes[i].value
                   for i in range(o_units)]
            delta[-1] = [(o_nodes[i].value) * (1 - o_nodes[i].value) *
                         (err[i]) for i in range(o_units)]

            # Backward pass
            h_layers = n_layers - 2
            for i in range(h_layers, 0, -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net[i+1]
                # weights from each ith layer node to each i + 1th layer node
                w = [[node.weights[k] for node in nx_layer]
                     for k in range(h_units)]

                delta[i] = [(layer[j].value) * (1 - layer[j].value) *
                            dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]

            #  Update weights
            for i in range(1, n_layers):
                layer = net[i]
                inc = [node.value for node in net[i-1]]
                units = len(layer)
                for j in range(units):
                    layer[j].weights = vector_add(layer[j].weights,
                                                  scalar_vector_product(
                                                  learning_rate * delta[i][j], inc))

    return net


def PerceptronLearner(dataset, learning_rate=0.01, epochs=100):
    """Logistic Regression, NO hidden layer"""
    i_units = len(dataset.inputs)
    o_units = 1  # As of now, dataset.target gives only one index.
    hidden_layer_sizes = []
    raw_net = network(i_units, hidden_layer_sizes, o_units)
    learned_net = BackPropagationLearner(dataset, raw_net, learning_rate, epochs)

    def predict(example):
        # Input nodes
        i_nodes = learned_net[0]

        # Activate input layer
        for v, n in zip(example, i_nodes):
            n.value = v

        # Forward pass
        for layer in learned_net[1:]:
            for node in layer:
                inc = [n.value for n in node.inputs]
                in_val = dotproduct(inc, node.weights)
                node.value = node.activation(in_val)

        # Hypothesis
        o_nodes = learned_net[-1]
        pred = [o_nodes[i].value for i in range(o_units)]
        return 1 if pred[0] >= 0.5 else 0

    return predict
# ______________________________________________________________________________


def LinearLearner(dataset, learning_rate=0.01, epochs=100):
    """Define with learner = LinearLearner(data); infer with learner(x)."""
    idx_i = dataset.inputs
    idx_t = dataset.target  # As of now, dataset.target gives only one index.
    examples = dataset.examples

    # X transpose
    X_col = [dataset.values[i] for i in idx_i]  # vertical columns of X

    # Add dummy
    ones = [1 for _ in range(len(examples))]
    X_col = ones + X_col

    # Initialize random weigts
    w = [random.randrange(-0.5, 0.5) for _ in range(len(idx_i) + 1)]

    for epoch in range(epochs):
        err = []
        # Pass over all examples
        for example in examples:
            x = [1] + example
            y = dotproduct(w, x)
            t = example[idx_t]
            err.append(t - y)

        # update weights
        for i in range(len(w)):
            w[i] = w[i] - learning_rate * dotproduct(err, X_col[i])

    def predict(example):
        x = [1] + example
        return dotproduct(w, x)
    return predict

# ______________________________________________________________________________


def EnsembleLearner(learners):
    """Given a list of learning algorithms, have them vote."""
    def train(dataset):
        predictors = [learner(dataset) for learner in learners]

        def predict(example):
            return mode(predictor(example) for predictor in predictors)
        return predict
    return train

# ______________________________________________________________________________


def AdaBoost(L, K):
    """[Figure 18.34]"""
    def train(dataset):
        examples, target = dataset.examples, dataset.target
        N = len(examples)
        epsilon = 1. / (2 * N)
        w = [1. / N] * N
        h, z = [], []
        for k in range(K):
            h_k = L(dataset, w)
            h.append(h_k)
            error = sum(weight for example, weight in zip(examples, w)
                        if example[target] != h_k(example))
            # Avoid divide-by-0 from either 0% or 100% error rates:
            error = clip(error, epsilon, 1 - epsilon)
            for j, example in enumerate(examples):
                if example[target] == h_k(example):
                    w[j] *= error / (1. - error)
            w = normalize(w)
            z.append(math.log((1. - error) / error))
        return WeightedMajority(h, z)
    return train


def WeightedMajority(predictors, weights):
    """Return a predictor that takes a weighted vote."""
    def predict(example):
        return weighted_mode((predictor(example) for predictor in predictors),
                             weights)
    return predict


def weighted_mode(values, weights):
    """Return the value with the greatest total weight.
    >>> weighted_mode('abbaa', [1,2,3,1,2])
    'b'
    """
    totals = defaultdict(int)
    for v, w in zip(values, weights):
        totals[v] += w
    return max(list(totals.keys()), key=totals.get)

# _____________________________________________________________________________
# Adapting an unweighted learner for AdaBoost


def WeightedLearner(unweighted_learner):
    """Given a learner that takes just an unweighted dataset, return
    one that takes also a weight for each example. [p. 749 footnote 14]"""
    def train(dataset, weights):
        return unweighted_learner(replicated_dataset(dataset, weights))
    return train


def replicated_dataset(dataset, weights, n=None):
    """Copy dataset, replicating each example in proportion to its weight."""
    n = n or len(dataset.examples)
    result = copy.copy(dataset)
    result.examples = weighted_replicate(dataset.examples, weights, n)
    return result


def weighted_replicate(seq, weights, n):
    """Return n selections from seq, with the count of each element of
    seq proportional to the corresponding weight (filling in fractions
    randomly).
    >>> weighted_replicate('ABC', [1,2,1], 4)
    ['A', 'B', 'B', 'C']
    """
    assert len(seq) == len(weights)
    weights = normalize(weights)
    wholes = [int(w * n) for w in weights]
    fractions = [(w * n) % 1 for w in weights]
    return (flatten([x] * nx for x, nx in zip(seq, wholes)) +
            weighted_sample_with_replacement(n - sum(wholes),seq, fractions, ))


def flatten(seqs): return sum(seqs, [])

# _____________________________________________________________________________
# Functions for testing learners on examples


def test(predict, dataset, examples=None, verbose=0):
    """Return the proportion of the examples that are NOT correctly predicted."""
    if examples is None:
        examples = dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0.0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1
            if verbose >= 2:
                print('   OK: got {} for {}'.format(desired, example))
        elif verbose:
            print('WRONG: got {}, expected {} for {}'.format(
                output, desired, example))
    return 1 - (right / len(examples))


def train_and_test(dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder."""
    start = int(start)
    end = int(end)
    examples = dataset.examples
    train = examples[:start] + examples[end:]
    val = examples[start:end]
    return train, val


def cross_validation(learner, size, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials>1, average over several shuffles.
    Returns Training error, Validataion error"""
    if k is None:
        k = len(dataset.examples)
    if trials > 1:
        trial_errT = 0
        trial_errV = 0
        for t in range(trials):
            errT, errV = cross_validation(learner, size, dataset,
                                          k=10, trials=1)
            trial_errT += errT
            trial_errV += errV
        return trial_errT / trials, trial_errV / trials
    else:
        fold_errT = 0
        fold_errV = 0
        n = len(dataset.examples)
        examples = dataset.examples
        for fold in range(k):
            random.shuffle(dataset.examples)
            train_data, val_data = train_and_test(dataset, fold * (n / k),
                                                  (fold + 1) * (n / k))
            dataset.examples = train_data
            h = learner(dataset, size)
            fold_errT += test(h, dataset, train_data)
            fold_errV += test(h, dataset, val_data)
            # Reverting back to original once test is completed
            dataset.examples = examples
        return fold_errT / k, fold_errV / k


def cross_validation_wrapper(learner, dataset, k=10, trials=1):
    """[Fig 18.8]
    Return the optimal value of size having minimum error
    on validataion set.
    err_train: A training error array, indexed by size
    err_val: A validataion error array, indexed by size
    """
    err_val = []
    err_train = []
    size = 1
    while True:
        errT, errV = cross_validation(learner, size, dataset, k)
        # Check for convergence provided err_val is not empty
        if (err_val and isclose(err_val[-1], errV, rel_tol=1e-6)):
            best_size = size
            return learner(dataset, best_size)

        err_val.append(errV)
        err_train.append(errT)
        print(err_val)
        size += 1


def leave_one_out(learner, dataset):
    """Leave one out cross-validation over the dataset."""
    return cross_validation(learner, size, dataset, k=len(dataset.examples))


def learningcurve(learner, dataset, trials=10, sizes=None):
    if sizes is None:
        sizes = list(range(2, len(dataset.examples) - 10, 2))

    def score(learner, size):
        random.shuffle(dataset.examples)
        return train_and_test(learner, dataset, 0, size)
    return [(size, mean([score(learner, size) for t in range(trials)]))
            for size in sizes]

# ______________________________________________________________________________
# The rest of this file gives datasets for machine learning problems.

orings = DataSet(name='orings', target='Distressed',
                 attrnames="Rings Distressed Temp Pressure Flightnum")


zoo = DataSet(name='zoo', target='type', exclude=['name'],
              attrnames="name hair feathers eggs milk airborne aquatic " +
              "predator toothed backbone breathes venomous fins legs tail " +
              "domestic catsize type")


iris = DataSet(name="iris", target="class",
               attrnames="sepal-len sepal-width petal-len petal-width class")

# ______________________________________________________________________________
# The Restaurant example from [Figure 18.2]


def RestaurantDataSet(examples=None):
    """Build a DataSet of Restaurant waiting examples. [Figure 18.3]"""
    return DataSet(name='restaurant', target='Wait', examples=examples,
                   attrnames='Alternate Bar Fri/Sat Hungry Patrons Price ' +
                   'Raining Reservation Type WaitEstimate Wait')

restaurant = RestaurantDataSet()


def T(attrname, branches):
    branches = {value: (child if isinstance(child, DecisionFork)
                        else DecisionLeaf(child))
                for value, child in branches.items()}
    return DecisionFork(restaurant.attrnum(attrname), attrname, branches)

""" [Figure 18.2]
A decision tree for deciding whether to wait for a table at a hotel.
"""

waiting_decision_tree = T('Patrons',
               {'None': 'No', 'Some': 'Yes', 'Full':
                T('WaitEstimate',
                  {'>60': 'No', '0-10': 'Yes',
                   '30-60':
                   T('Alternate', {'No':
                                   T('Reservation', {'Yes': 'Yes', 'No':
                                                     T('Bar', {'No': 'No',
                                                               'Yes': 'Yes'
                                                               })}),
                                   'Yes':
                                   T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                   '10-30':
                   T('Hungry', {'No': 'Yes', 'Yes':
                                T('Alternate',
                                  {'No': 'Yes', 'Yes':
                                   T('Raining', {'No': 'No', 'Yes': 'Yes'})
                                   })})})})


def SyntheticRestaurant(n=20):
    """Generate a DataSet with n examples."""
    def gen():
        example = list(map(random.choice, restaurant.values))
        example[restaurant.target] = waiting_decision_tree(example)
        return example
    return RestaurantDataSet([gen() for i in range(n)])

# ______________________________________________________________________________
# Artificial, generated datasets.


def Majority(k, n):
    """Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(int(sum(bits) > k / 2))
        examples.append(bits)
    return DataSet(name="majority", examples=examples)


def Parity(k, n, name="parity"):
    """Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) % 2)
        examples.append(bits)
    return DataSet(name=name, examples=examples)


def Xor(n):
    """Return a DataSet with n examples of 2-input xor."""
    return Parity(2, n, name="xor")


def ContinuousXor(n):
    "2 inputs are chosen uniformly from (0.0 .. 2.0]; output is xor of ints."
    examples = []
    for i in range(n):
        x, y = [random.uniform(0.0, 2.0) for i in '12']
        examples.append([x, y, int(x) != int(y)])
    return DataSet(name="continuous xor", examples=examples)

# ______________________________________________________________________________


def compare(algorithms=[PluralityLearner, NaiveBayesLearner,
                        NearestNeighborLearner, DecisionTreeLearner],
            datasets=[iris, orings, zoo, restaurant, SyntheticRestaurant(20),
                      Majority(7, 100), Parity(7, 100), Xor(100)],
            k=10, trials=1):
    """Compare various learners on various datasets using cross-validation.
    Print results as a table."""
    print_table([[a.__name__.replace('Learner', '')] +
                 [cross_validation(a, d, k, trials) for d in datasets]
                 for a in algorithms],
                header=[''] + [d.name[0:7] for d in datasets], numfmt='%.2f')
