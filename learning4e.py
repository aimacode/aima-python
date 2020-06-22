"""Learning from examples (Chapters 18)"""

import copy
from collections import defaultdict
from statistics import stdev

from qpsolvers import solve_qp

from deep_learning4e import Sigmoid
from probabilistic_learning import NaiveBayesLearner
from utils4e import *


class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + '.csv').read())
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
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
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)][:-1]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Finds the means and standard deviations of self.dataset.
        means     : a dictionary for each class/target. Holds a list of the means
                    of the features for the class.
        deviations: a dictionary for each class/target. Holds a list of the sample
                    standard deviations of the features for the class.
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations fo the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])

        return means, deviations

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))


def parse_csv(input, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def err_ratio(learner, dataset, examples=None):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = learner.predict(dataset.sanitize(example))
        if np.allclose(output, desired):
            right += 1
    return 1 - (right / len(examples))


def grade_learner(learner, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    return mean(int(learner.predict(X) == y) for X, y in tests)


def train_test_split(dataset, start=None, end=None, test_split=None):
    """
    If you are giving 'start' and 'end' as parameters,
    then it will return the testing set from index 'start' to 'end'
    and the rest for training.
    If you give 'test_split' as a parameter then it will return
    test_split * 100% as the testing set and the rest as
    training set.
    """
    examples = dataset.examples
    if test_split is None:
        train = examples[:start] + examples[end:]
        val = examples[start:end]
    else:
        total_size = len(examples)
        val_size = int(total_size * test_split)
        train_size = total_size - val_size
        train = examples[:train_size]
        val = examples[train_size:total_size]

    return train, val


def model_selection(learner, dataset, k=10, trials=1):
    """
    [Figure 18.8]
    Return the optimal value of size having minimum error on validation set.
    err: a validation error array, indexed by size
    """
    errs = []
    size = 1
    while True:
        err = cross_validation(learner, dataset, size, k, trials)
        # check for convergence provided err_val is not empty
        if err and not np.isclose(err[-1], err, rtol=1e-6):
            best_size = 0
            min_val = np.inf
            i = 0
            while i < size:
                if errs[i] < min_val:
                    min_val = errs[i]
                    best_size = i
                i += 1
            return learner(dataset, best_size)
        errs.append(err)
        size += 1


def cross_validation(learner, dataset, size=None, k=10, trials=1):
    """
    Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials > 1, average over several shuffles.
    Returns Training error
    """
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errs = 0
        for t in range(trials):
            errs = cross_validation(learner, dataset, size, k, trials)
            trial_errs += errs
        return trial_errs / trials
    else:
        fold_errs = 0
        n = len(dataset.examples)
        examples = dataset.examples
        random.shuffle(dataset.examples)
        for fold in range(k):
            train_data, val_data = train_test_split(dataset, fold * (n // k), (fold + 1) * (n // k))
            dataset.examples = train_data
            h = learner(dataset, size)
            fold_errs += err_ratio(h, dataset, train_data)
            # reverting back to original once test is completed
            dataset.examples = examples
        return fold_errs / k


def leave_one_out(learner, dataset, size=None):
    """Leave one out cross-validation over the dataset."""
    return cross_validation(learner, dataset, size, len(dataset.examples))


def learning_curve(learner, dataset, trials=10, sizes=None):
    if sizes is None:
        sizes = list(range(2, len(dataset.examples) - trials, 2))

    def score(learner, size):
        random.shuffle(dataset.examples)
        return cross_validation(learner, dataset, size, trials)

    return [(size, mean([score(learner, size) for _ in range(trials)])) for size in sizes]


class PluralityLearner:
    """
    A very dumb algorithm: always pick the result that was most popular
    in the training data. Makes a baseline for comparison.
    """

    def __init__(self, dataset):
        self.most_popular = mode([e[dataset.target] for e in dataset.examples])

    def predict(self, example):
        """Always return same result: the most popular from the training set."""
        return self.most_popular


class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attr_name
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(self.attr, self.attr_name, self.branches)


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)


class DecisionTreeLearner:
    """[Figure 18.5]"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def decision_tree_learning(self, examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        if self.all_same_class(examples):
            return DecisionLeaf(examples[0][self.dataset.target])
        if len(attrs) == 0:
            return self.plurality_value(examples)
        A = self.choose_attribute(attrs, examples)
        tree = DecisionFork(A, self.dataset.attr_names[A], self.plurality_value(examples))
        for (v_k, exs) in self.split_by(A, examples):
            subtree = self.decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(self, examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(self.dataset.values[self.dataset.target],
                                    key=lambda v: self.count(self.dataset.target, v, examples))
        return DecisionLeaf(popular)

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.dataset.target]
        return all(e[self.dataset.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: self.information_gain(a, examples))

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def I(examples):
            return information_content([self.count(self.dataset.target, v, examples)
                                        for v in self.dataset.values[self.dataset.target]])

        n = len(examples)
        remainder = sum((len(examples_i) / n) * I(examples_i)
                        for (v, examples_i) in self.split_by(attr, examples))
        return I(examples) - remainder

    def split_by(self, attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in self.dataset.values[attr]]

    def predict(self, x):
        return self.tree(x)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(remove_all(0, values))
    return sum(-p * np.log2(p) for p in probabilities)


class DecisionListLearner:
    """
    [Figure 18.11]
    A decision list implemented as a list of (test, value) pairs.
    """

    def __init__(self, dataset):
        self.predict.decision_list = self.decision_list_learning(set(dataset.examples))

    def decision_list_learning(self, examples):
        if not examples:
            return [(True, False)]
        t, o, examples_t = self.find_examples(examples)
        if not t:
            raise Exception
        return [(t, o)] + self.decision_list_learning(examples - examples_t)

    def find_examples(self, examples):
        """
        Find a set of examples that all have the same outcome under
        some test. Return a tuple of the test, outcome, and examples.
        """
        raise NotImplementedError

    def passes(self, example, test):
        """Does the example pass the test?"""
        raise NotImplementedError

    def predict(self, example):
        """Predict the outcome for the first passing test."""
        for test, outcome in self.predict.decision_list:
            if self.passes(example, test):
                return outcome


class NearestNeighborLearner:
    """k-NearestNeighbor: the k nearest neighbors vote."""

    def __init__(self, dataset, k=1):
        self.dataset = dataset
        self.k = k

    def predict(self, example):
        """Find the k closest items, and have them vote for the best."""
        best = heapq.nsmallest(self.k, ((self.dataset.distance(e, example), e) for e in self.dataset.examples))
        return mode(e[self.dataset.target] for (d, e) in best)


class SVC:

    def __init__(self, kernel=linear_kernel, C=1.0, verbose=False):
        self.kernel = kernel
        self.C = C  # hyper-parameter
        self.sv_idx, self.sv, self.sv_y = np.zeros(0), np.zeros(0), np.zeros(0)
        self.alphas = np.zeros(0)
        self.w = None
        self.b = 0.0  # intercept
        self.verbose = verbose

    def fit(self, X, y):
        """
        Trains the model by solving a quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        # In QP formulation (dual): m variables, 2m+1 constraints (1 equation, 2m inequations)
        self.solve_qp(X, y)
        sv = self.alphas > 1e-5
        self.sv_idx = np.arange(len(self.alphas))[sv]
        self.sv, self.sv_y, self.alphas = X[sv], y[sv], self.alphas[sv]

        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.sv)

        for n in range(len(self.alphas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * self.K[self.sv_idx[n], sv])
        self.b /= len(self.alphas)
        return self

    def solve_qp(self, X, y):
        """
        Solves a quadratic programming problem. In QP formulation (dual):
        m variables, 2m+1 constraints (1 equation, 2m inequations).
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        m = len(y)  # m = n_samples
        self.K = self.kernel(X)  # gram matrix
        P = self.K * np.outer(y, y)
        q = -np.ones(m)
        lb = np.zeros(m)  # lower bounds
        ub = np.ones(m) * self.C  # upper bounds
        A = y.astype(np.float64)  # equality matrix
        b = np.zeros(1)  # equality vector
        self.alphas = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='cvxopt',
                               sym_proj=True, verbose=self.verbose)

    def predict_score(self, X):
        """
        Predicts the score for a given example.
        """
        if self.w is None:
            return np.dot(self.alphas * self.sv_y, self.kernel(self.sv, X)) + self.b
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.sign(self.predict_score(X))


class SVR:

    def __init__(self, kernel=linear_kernel, C=1.0, epsilon=0.1, verbose=False):
        self.kernel = kernel
        self.C = C  # hyper-parameter
        self.epsilon = epsilon  # epsilon insensitive loss value
        self.sv_idx, self.sv = np.zeros(0), np.zeros(0)
        self.alphas_p, self.alphas_n = np.zeros(0), np.zeros(0)
        self.w = None
        self.b = 0.0  # intercept
        self.verbose = verbose

    def fit(self, X, y):
        """
        Trains the model by solving a quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        # In QP formulation (dual): m variables, 2m+1 constraints (1 equation, 2m inequations)
        self.solve_qp(X, y)

        sv = np.logical_or(self.alphas_p > 1e-5, self.alphas_n > 1e-5)
        self.sv_idx = np.arange(len(self.alphas_p))[sv]
        self.sv, sv_y = X[sv], y[sv]
        self.alphas_p, self.alphas_n = self.alphas_p[sv], self.alphas_n[sv]

        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas_p - self.alphas_n, self.sv)

        for n in range(len(self.alphas_p)):
            self.b += sv_y[n]
            self.b -= np.sum((self.alphas_p - self.alphas_n) * self.K[self.sv_idx[n], sv])
        self.b -= self.epsilon
        self.b /= len(self.alphas_p)

        return self

    def solve_qp(self, X, y):
        """
        Solves a quadratic programming problem. In QP formulation (dual):
        m variables, 2m+1 constraints (1 equation, 2m inequations).
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        m = len(y)  # m = n_samples
        self.K = self.kernel(X)  # gram matrix
        P = np.vstack((np.hstack((self.K, -self.K)),  # alphas_p, alphas_n
                       np.hstack((-self.K, self.K))))  # alphas_n, alphas_p
        q = np.hstack((-y, y)) + self.epsilon
        lb = np.zeros(2 * m)  # lower bounds
        ub = np.ones(2 * m) * self.C  # upper bounds
        A = np.hstack((np.ones(m), -np.ones(m)))  # equality matrix
        b = np.zeros(1)  # equality vector
        alphas = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='cvxopt',
                          sym_proj=True, verbose=self.verbose)
        self.alphas_p = alphas[:m]
        self.alphas_n = alphas[m:]

    def predict(self, X):
        if self.kernel != linear_kernel:
            return np.dot(self.alphas_p - self.alphas_n, self.kernel(self.sv, X)) + self.b
        return np.dot(X, self.w) + self.b


class MultiClassLearner:

    def __init__(self, clf, decision_function='ovr'):
        self.clf = clf
        self.decision_function = decision_function
        self.n_class, self.classifiers = 0, []

    def fit(self, X, y):
        """
        Trains n_class or n_class * (n_class - 1) / 2 classifiers
        according to the training method, ovr or ovo respectively.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        :return: array of classifiers
        """
        labels = np.unique(y)
        self.n_class = len(labels)
        if self.decision_function == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                self.clf.fit(X, y1)
                self.classifiers.append(copy.deepcopy(self.clf))
        elif self.decision_function == 'ovo':  # use one-vs-one method
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    X1, y1 = np.r_[X[neg_id], X[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    self.clf.fit(X1, y1)
                    self.classifiers.append(copy.deepcopy(self.clf))
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")
        return self

    def predict(self, X):
        """
        Predicts the class of a given example according to the training method.
        """
        n_samples = len(X)
        if self.decision_function == 'ovr':  # one-vs-rest method
            assert len(self.classifiers) == self.n_class
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(X)
            return np.argmax(score, axis=1)
        elif self.decision_function == 'ovo':  # use one-vs-one method
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    res = self.classifiers[clf_id].predict(X)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")


def LinearLearner(dataset, learning_rate=0.01, epochs=100):
    """
    [Section 18.6.3]
    Linear classifier with hard threshold.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # X transpose
    X_col = [dataset.values[i] for i in idx_i]  # vertical columns of X

    # add dummy
    ones = [1 for _ in range(len(examples))]
    X_col = [ones] + X_col

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = random_weights(min_value=-0.5, max_value=0.5, num_weights=num_weights)

    for epoch in range(epochs):
        err = []
        # pass over all examples
        for example in examples:
            x = [1] + example
            y = np.dot(w, x)
            t = example[idx_t]
            err.append(t - y)

        # update weights
        for i in range(len(w)):
            w[i] = w[i] + learning_rate * (np.dot(err, X_col[i]) / num_examples)

    def predict(example):
        x = [1] + example
        return np.dot(w, x)

    return predict


def LogisticLinearLeaner(dataset, learning_rate=0.01, epochs=100):
    """
    [Section 18.6.4]
    Linear classifier with logistic regression.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # X transpose
    X_col = [dataset.values[i] for i in idx_i]  # vertical columns of X

    # add dummy
    ones = [1 for _ in range(len(examples))]
    X_col = [ones] + X_col

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = random_weights(min_value=-0.5, max_value=0.5, num_weights=num_weights)

    for epoch in range(epochs):
        err = []
        h = []
        # pass over all examples
        for example in examples:
            x = [1] + example
            y = Sigmoid()(np.dot(w, x))
            h.append(Sigmoid().derivative(y))
            t = example[idx_t]
            err.append(t - y)

        # update weights
        for i in range(len(w)):
            buffer = [x * y for x, y in zip(err, h)]
            w[i] = w[i] + learning_rate * (np.dot(buffer, X_col[i]) / num_examples)

    def predict(example):
        x = [1] + example
        return Sigmoid()(np.dot(w, x))

    return predict


class EnsembleLearner:
    """Given a list of learning algorithms, have them vote."""

    def __init__(self, learners):
        self.learners = learners

    def train(self, dataset):
        self.predictors = [learner(dataset) for learner in self.learners]

    def predict(self, example):
        return mode(predictor.predict(example) for predictor in self.predictors)


def ada_boost(dataset, L, K):
    """[Figure 18.34]"""

    examples, target = dataset.examples, dataset.target
    n = len(examples)
    eps = 1 / (2 * n)
    w = [1 / n] * n
    h, z = [], []
    for k in range(K):
        h_k = L(dataset, w)
        h.append(h_k)
        error = sum(weight for example, weight in zip(examples, w) if example[target] != h_k.predict(example[:-1]))
        # avoid divide-by-0 from either 0% or 100% error rates
        error = np.clip(error, eps, 1 - eps)
        for j, example in enumerate(examples):
            if example[target] == h_k.predict(example[:-1]):
                w[j] *= error / (1 - error)
        w = normalize(w)
        z.append(np.log((1 - error) / error))
    return weighted_majority(h, z)


class weighted_majority:
    """Return a predictor that takes a weighted vote."""

    def __init__(self, predictors, weights):
        self.predictors = predictors
        self.weights = weights

    def predict(self, example):
        return weighted_mode((predictor.predict(example) for predictor in self.predictors), self.weights)


def weighted_mode(values, weights):
    """
    Return the value with the greatest total weight.
    >>> weighted_mode('abbaa', [1, 2, 3, 1, 2])
    'b'
    """
    totals = defaultdict(int)
    for v, w in zip(values, weights):
        totals[v] += w
    return max(totals, key=totals.__getitem__)


class RandomForest:
    """An ensemble of Decision Trees trained using bagging and feature bagging."""

    def __init__(self, dataset, n=5):
        self.dataset = dataset
        self.n = n
        self.predictors = [DecisionTreeLearner(DataSet(examples=self.data_bagging(), attrs=self.dataset.attrs,
                                                       attr_names=self.dataset.attr_names, target=self.dataset.target,
                                                       inputs=self.feature_bagging())) for _ in range(self.n)]

    def data_bagging(self, m=0):
        """Sample m examples with replacement"""
        n = len(self.dataset.examples)
        return weighted_sample_with_replacement(m or n, self.dataset.examples, [1] * n)

    def feature_bagging(self, p=0.7):
        """Feature bagging with probability p to retain an attribute"""
        inputs = [i for i in self.dataset.inputs if probability(p)]
        return inputs or self.dataset.inputs

    def predict(self, example):
        return mode(predictor.predict(example) for predictor in self.predictors)


def WeightedLearner(unweighted_learner):
    """
    [Page 749 footnote 14]
    Given a learner that takes just an unweighted dataset, return
    one that takes also a weight for each example.
    """

    def train(dataset, weights):
        dataset = replicated_dataset(dataset, weights)
        n_samples, n_features = len(dataset.examples), dataset.target
        X, y = (np.array([x[:n_features] for x in dataset.examples]),
                np.array([x[n_features] for x in dataset.examples]))
        return unweighted_learner.fit(X, y)

    return train


def replicated_dataset(dataset, weights, n=None):
    """Copy dataset, replicating each example in proportion to its weight."""
    n = n or len(dataset.examples)
    result = copy.copy(dataset)
    result.examples = weighted_replicate(dataset.examples, weights, n)
    return result


def weighted_replicate(seq, weights, n):
    """
    Return n selections from seq, with the count of each element of
    seq proportional to the corresponding weight (filling in fractions
    randomly).
    >>> weighted_replicate('ABC', [1, 2, 1], 4)
    ['A', 'B', 'B', 'C']
    """
    assert len(seq) == len(weights)
    weights = normalize(weights)
    wholes = [int(w * n) for w in weights]
    fractions = [(w * n) % 1 for w in weights]
    return (flatten([x] * nx for x, nx in zip(seq, wholes)) +
            weighted_sample_with_replacement(n - sum(wholes), seq, fractions))


# metrics

def accuracy_score(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.equal(y_pred, y_true))


def r2_score(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 1. - (np.sum(np.square(y_pred - y_true)) /  # sum of square of residuals
                 np.sum(np.square(y_true - np.mean(y_true))))  # total sum of squares


# datasets

orings = DataSet(name='orings', target='Distressed', attr_names='Rings Distressed Temp Pressure Flightnum')

zoo = DataSet(name='zoo', target='type', exclude=['name'],
              attr_names='name hair feathers eggs milk airborne aquatic predator toothed backbone '
                         'breathes venomous fins legs tail domestic catsize type')

iris = DataSet(name='iris', target='class', attr_names='sepal-len sepal-width petal-len petal-width class')


def RestaurantDataSet(examples=None):
    """
    [Figure 18.3]
    Build a DataSet of Restaurant waiting examples.
    """
    return DataSet(name='restaurant', target='Wait', examples=examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Type WaitEstimate Wait')


restaurant = RestaurantDataSet()


def T(attr_name, branches):
    branches = {value: (child if isinstance(child, DecisionFork) else DecisionLeaf(child))
                for value, child in branches.items()}
    return DecisionFork(restaurant.attr_num(attr_name), attr_name, print, branches)


""" 
[Figure 18.2]
A decision tree for deciding whether to wait for a table at a hotel.
"""

waiting_decision_tree = T('Patrons',
                          {'None': 'No', 'Some': 'Yes',
                           'Full': T('WaitEstimate',
                                     {'>60': 'No', '0-10': 'Yes',
                                      '30-60': T('Alternate',
                                                 {'No': T('Reservation',
                                                          {'Yes': 'Yes',
                                                           'No': T('Bar', {'No': 'No',
                                                                           'Yes': 'Yes'})}),
                                                  'Yes': T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                                      '10-30': T('Hungry',
                                                 {'No': 'Yes',
                                                  'Yes': T('Alternate',
                                                           {'No': 'Yes',
                                                            'Yes': T('Raining',
                                                                     {'No': 'No',
                                                                      'Yes': 'Yes'})})})})})


def SyntheticRestaurant(n=20):
    """Generate a DataSet with n examples."""

    def gen():
        example = list(map(random.choice, restaurant.values))
        example[restaurant.target] = waiting_decision_tree(example)
        return example

    return RestaurantDataSet([gen() for _ in range(n)])


def Majority(k, n):
    """
    Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0.
    """
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for _ in range(k)]
        bits.append(int(sum(bits) > k / 2))
        examples.append(bits)
    return DataSet(name='majority', examples=examples)


def Parity(k, n, name='parity'):
    """
    Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0.
    """
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for _ in range(k)]
        bits.append(sum(bits) % 2)
        examples.append(bits)
    return DataSet(name=name, examples=examples)


def Xor(n):
    """Return a DataSet with n examples of 2-input xor."""
    return Parity(2, n, name='xor')


def ContinuousXor(n):
    """2 inputs are chosen uniformly from (0.0 .. 2.0]; output is xor of ints."""
    examples = []
    for i in range(n):
        x, y = [random.uniform(0.0, 2.0) for _ in '12']
        examples.append([x, y, x != y])
    return DataSet(name='continuous xor', examples=examples)


def compare(algorithms=None, datasets=None, k=10, trials=1):
    """
    Compare various learners on various datasets using cross-validation.
    Print results as a table.
    """
    # default list of algorithms
    algorithms = algorithms or [PluralityLearner, NaiveBayesLearner, NearestNeighborLearner, DecisionTreeLearner]

    # default list of datasets
    datasets = datasets or [iris, orings, zoo, restaurant, SyntheticRestaurant(20),
                            Majority(7, 100), Parity(7, 100), Xor(100)]

    print_table([[a.__name__.replace('Learner', '')] + [cross_validation(a, d, k=k, trials=trials) for d in datasets]
                 for a in algorithms], header=[''] + [d.name[0:7] for d in datasets], numfmt='%.2f')
