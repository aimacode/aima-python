"""Learning probabilistic models. (Chapters 20)"""

import heapq

from utils import weighted_sampler, product, gaussian


class CountingProbDist:
    """
    A probability distribution formed by observing and counting examples.
    If p is an instance of this class and o is an observed value, then
    there are 3 main operations:
    p.add(o) increments the count for observation o by 1.
    p.sample() returns a random element from the distribution.
    p[o] returns the probability for o (as in a regular ProbDist).
    """

    def __init__(self, observations=None, default=0):
        """
        Create a distribution, and optionally add in some observations.
        By default this is an unsmoothed distribution, but saying default=1,
        for example, gives you add-one smoothing.
        """
        if observations is None:
            observations = []
        self.dictionary = {}
        self.n_obs = 0
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
        """
        Include o among the possible observations, whether or not
        it's been observed yet.
        """
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
            self.sampler = weighted_sampler(list(self.dictionary.keys()), list(self.dictionary.values()))
        return self.sampler()


def NaiveBayesLearner(dataset, continuous=True, simple=False):
    if simple:
        return NaiveBayesSimple(dataset)
    if continuous:
        return NaiveBayesContinuous(dataset)
    else:
        return NaiveBayesDiscrete(dataset)


def NaiveBayesSimple(distribution):
    """
    A simple naive bayes classifier that takes as input a dictionary of
    CountingProbDist objects and classifies items according to these distributions.
    The input dictionary is in the following form:
        (ClassName, ClassProb): CountingProbDist
    """
    target_dist = {c_name: prob for c_name, prob in distribution.keys()}
    attr_dists = {c_name: count_prob for (c_name, _), count_prob in distribution.items()}

    def predict(example):
        """Predict the target value for example. Calculate probabilities for each
        class and pick the max."""

        def class_probability(target_val):
            attr_dist = attr_dists[target_val]
            return target_dist[target_val] * product(attr_dist[a] for a in example)

        return max(target_dist.keys(), key=class_probability)

    return predict


def NaiveBayesDiscrete(dataset):
    """
    Just count how many times each value of each input attribute
    occurs, conditional on the target value. Count the different
    target values too.
    """

    target_vals = dataset.values[dataset.target]
    target_dist = CountingProbDist(target_vals)
    attr_dists = {(gv, attr): CountingProbDist(dataset.values[attr]) for gv in target_vals for attr in dataset.inputs}
    for example in dataset.examples:
        target_val = example[dataset.target]
        target_dist.add(target_val)
        for attr in dataset.inputs:
            attr_dists[target_val, attr].add(example[attr])

    def predict(example):
        """
        Predict the target value for example. Consider each possible value,
        and pick the most likely by looking at each attribute independently.
        """

        def class_probability(target_val):
            return (target_dist[target_val] * product(attr_dists[target_val, attr][example[attr]]
                                                      for attr in dataset.inputs))

        return max(target_vals, key=class_probability)

    return predict


def NaiveBayesContinuous(dataset):
    """
    Count how many times each target value occurs.
    Also, find the means and deviations of input attribute values for each target value.
    """
    means, deviations = dataset.find_means_and_deviations()

    target_vals = dataset.values[dataset.target]
    target_dist = CountingProbDist(target_vals)

    def predict(example):
        """Predict the target value for example. Consider each possible value,
        and pick the most likely by looking at each attribute independently."""

        def class_probability(target_val):
            prob = target_dist[target_val]
            for attr in dataset.inputs:
                prob *= gaussian(means[target_val][attr], deviations[target_val][attr], example[attr])
            return prob

        return max(target_vals, key=class_probability)

    return predict
