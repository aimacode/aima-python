"""Learn to estimate functions  from examples. (Chapters 18-20)"""

from utils import *
import agents, random, operator

#______________________________________________________________________________

class DataSet:
    """A data set for a machine learning problem.  It has the following fields:

    d.examples    A list of examples.  Each one is a list of attribute values.
    d.attrs       A list of integers to index into an example, so example[attr]
                  gives a value. Normally the same as range(len(d.examples)). 
    d.attrnames   Optional list of mnemonic names for corresponding attrs.
    d.target      The attribute that a learning algorithm will try to predict.
                  By default the final attribute.
    d.inputs      The list of attrs without the target.
    d.values      A list of lists, each sublist is the set of possible
                  values for the corresponding attribute. If None, it
                  is computed from the known examples by self.setproblem.
                  If not None, an erroneous value raises ValueError.
    d.name        Name of the data set (for output display only).
    d.source      URL or other source where the data came from.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, examples=None, attrs=None, target=-1, values=None,
                 attrnames=None, name='', source='',
                 inputs=None, exclude=(), doc=''):
        """Accepts any of DataSet's fields.  Examples can
        also be a string or file from which to parse examples using parse_csv.
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        update(self, name=name, source=source, values=values)
        # Initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(DataFile(name+'.csv').read())
        else:
            self.examples = examples
        map(self.check_example, self.examples)
        # Attrs are the indicies of examples, unless otherwise stated.
        if not attrs and self.examples:
            attrs = range(len(self.examples[0]))
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
        to not put use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""
        self.target = self.attrnum(target)
        exclude = map(self.attrnum, exclude)
        if inputs:
            self.inputs = removall(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs
                           if a is not self.target and a not in exclude]
        if not self.values:
            self.values = map(unique, zip(*self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value %s for attribute %s in %s' %
                                     (example[a], self.attrnames[a], example))

    def attrnum(self, attr):
        "Returns the number used for attr, which can be a name, or -n .. n."
        if attr < 0:
            return len(self.attrs) + attr
        elif isinstance(attr, str): 
            return self.attrnames.index(attr)
        else:
            return attr

    def sanitize(self, example):
       "Return a copy of example, with non-input attributes replaced by 0."
       return [i in self.inputs and example[i] for i in range(len(example))] 

    def __repr__(self):
        return '<DataSet(%s): %d examples, %d attributes>' % (
            self.name, len(self.examples), len(self.attrs))

#______________________________________________________________________________

def parse_csv(input, delim=','):
    r"""Input is a string consisting of lines, each line has comma-delimited 
    fields.  Convert this into a list of lists.  Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip() is not '']
    return [map(num_or_str, line.split(delim)) for line in lines]

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])


#______________________________________________________________________________

class Learner:
    """A Learner, or Learning Algorithm, can be trained with a dataset,
    and then asked to predict the target attribute of an example."""

    def train(self, dataset): 
        self.dataset = dataset

    def predict(self, example): 
        abstract

#______________________________________________________________________________

class MajorityLearner(Learner):
    """A very dumb algorithm: always pick the result that was most popular
    in the training data.  Makes a baseline for comparison."""

    def train(self, dataset):
        "Find the target value that appears most often."
        self.most_popular = mode([e[dataset.target] for e in dataset.examples])

    def predict(self, example):
        "Always return same result: the most popular from the training set."
        return self.most_popular

#______________________________________________________________________________

class NaiveBayesLearner(Learner):
    
    def train(self, dataset):
        """Just count the target/attr/val occurences.
        Count how many times each value of each attribute occurs.
        Store count in N[targetvalue][attr][val]. Let N[attr][None] be the
        sum over all vals."""
        N = {}
        ## Initialize to 0
        for gv in self.dataset.values[self.dataset.target]:
            N[gv] = {}
            for attr in self.dataset.attrs:
                N[gv][attr] = {}
                for val in self.dataset.values[attr]:
                    N[gv][attr][val] = 0
                    N[gv][attr][None] = 0
        ## Go thru examples
        for example in self.dataset.examples:
            Ngv = N[example[self.dataset.target]]
            for attr in self.dataset.attrs:
                Ngv[attr][example[attr]] += 1
                Ngv[attr][None] += 1
        self._N = N

    def N(self, targetval, attr, attrval):
       "Return the count in the training data of this combination."
       try:
          return self._N[targetval][attr][attrval]
       except KeyError:
          return 0

    def P(self, targetval, attr, attrval):
        """Smooth the raw counts to give a probability estimate.
        Estimate adds 1 to numerator and len(possible vals) to denominator."""
        return ((self.N(targetval, attr, attrval) + 1.0) /
                (self.N(targetval, attr, None) + len(self.dataset.values[attr])))

    def predict(self, example):
        """Predict the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently."""
        possible_values = self.dataset.values[self.dataset.target]
        def class_probability(targetval):
            return product([self.P(targetval, a, example[a])
                            for a in self.dataset.inputs], 1)
        return argmax(possible_values, class_probability)

#______________________________________________________________________________

class NearestNeighborLearner(Learner):

    def __init__(self, k=1):
        "k-NearestNeighbor: the k nearest neighbors vote."
        self.k = k

    def predict(self, example):
        """With k=1, find the point closest to example.
        With k>1, find k closest, and have them vote for the best."""
        if self.k == 1:
            neighbor = argmin(self.dataset.examples,
                              lambda e: self.distance(e, example))
            return neighbor[self.dataset.target]
        else:
            ## Maintain a sorted list of (distance, example) pairs.
            ## For very large k, a PriorityQueue would be better
            best = [] 
            for e in examples:
                d = self.distance(e, example)
                if len(best) < k: 
                    e.append((d, e))
                elif d < best[-1][0]:
                    best[-1] = (d, e)
                    best.sort()
            return mode([e[self.dataset.target] for (d, e) in best])

    def distance(self, e1, e2):
        return mean_boolean_error(e1, e2)

#______________________________________________________________________________

class DecisionTree:
    """A DecisionTree holds an attribute that is being tested, and a
    dict of {attrval: Tree} entries.  If Tree here is not a DecisionTree
    then it is the final classification of the example."""

    def __init__(self, attr, attrname=None, branches=None):
        "Initialize by saying what attribute this node tests."
        update(self, attr=attr, attrname=attrname or attr,
               branches=branches or {})

    def predict(self, example):
        "Given an example, use the tree to classify the example."
        child = self.branches[example[self.attr]]
        if isinstance(child, DecisionTree):
            return child.predict(example)
        else:
            return child

    def add(self, val, subtree):
        "Add a branch.  If self.attr = val, go to the given subtree."
        self.branches[val] = subtree
        return self

    def display(self, indent=0):
        name = self.attrname
        print 'Test', name
        for (val, subtree) in self.branches.items():
            print ' '*4*indent, name, '=', val, '==>',
            if isinstance(subtree, DecisionTree):
                subtree.display(indent+1)
            else:
                print 'RESULT = ', subtree

    def __repr__(self):
        return 'DecisionTree(%r, %r, %r)' % (
            self.attr, self.attrname, self.branches)

Yes, No = True, False
        
#______________________________________________________________________________

class DecisionTreeLearner(Learner):

    def predict(self, example):
        if isinstance(self.dt, DecisionTree):
            return self.dt.predict(example)
        else:
            return self.dt

    def train(self, dataset):
        self.dataset = dataset
        self.attrnames = dataset.attrnames
        self.dt = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def decision_tree_learning(self, examples, attrs, default=None):
        if len(examples) == 0:
            return default
        elif self.all_same_class(examples):
            return examples[0][self.dataset.target]
        elif  len(attrs) == 0:
            return self.majority_value(examples)
        else:
            best = self.choose_attribute(attrs, examples)
            tree = DecisionTree(best, self.attrnames[best])
            for (v, examples_i) in self.split_by(best, examples):
                subtree = self.decision_tree_learning(examples_i,
                  removeall(best, attrs), self.majority_value(examples))
                tree.add(v, subtree)
            return tree

    def choose_attribute(self, attrs, examples):
        "Choose the attribute with the highest information gain."
        return argmax(attrs, lambda a: self.information_gain(a, examples))

    def all_same_class(self, examples):
        "Are all these examples in the same target class?"
        target = self.dataset.target
        class0 = examples[0][target]
        for e in examples:
           if e[target] != class0: return False
        return True

    def majority_value(self, examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        g = self.dataset.target
        return argmax(self.dataset.values[g],
                      lambda v: self.count(g, v, examples))

    def count(self, attr, val, examples):
        return count_if(lambda e: e[attr] == val, examples)
    
    def information_gain(self, attr, examples):
        def I(examples):
            target = self.dataset.target
            return information_content([self.count(target, v, examples)
                                        for v in self.dataset.values[target]])
        N = float(len(examples))
        remainder = 0
        for (v, examples_i) in self.split_by(attr, examples):
            remainder += (len(examples_i) / N) * I(examples_i)
        return I(examples) - remainder

    def split_by(self, attr, examples=None):
        "Return a list of (val, examples) pairs for each val of attr."
        if examples == None:
            examples = self.dataset.examples
        return [(v, [e for e in examples if e[attr] == v])
                for v in self.dataset.values[attr]]
    
def information_content(values):
    "Number of bits to represent the probability distribution in values."
    # If the values do not sum to 1, normalize them to make them a Prob. Dist.
    values = removeall(0, values)
    s = float(sum(values))
    if s != 1.0: values = [v/s for v in values]
    return sum([- v * log2(v) for v in values])

#______________________________________________________________________________

### A decision list is implemented as a list of (test, value) pairs.

class DecisionListLearner(Learner):

    def train(self, dataset): 
        self.dataset = dataset 
        self.attrnames = dataset.attrnames 
        self.dl = self.decision_list_learning(Set(dataset.examples))

    def decision_list_learning(self, examples):
        """[Fig. 18.14]"""
        if not examples:
            return [(True, No)]
        t, o, examples_t = self.find_examples(examples)
        if not t:
            raise Failure
        return [(t, o)] + self.decision_list_learning(examples - examples_t)

    def find_examples(self, examples):
        """Find a set of examples that all have the same outcome under some test.
        Return a tuple of the test, outcome, and examples."""
        NotImplemented
#______________________________________________________________________________

class NeuralNetLearner(Learner):
   """Layered feed-forward network."""

   def __init__(self, sizes):
      self.activations = map(lambda n: [0.0 for i in range(n)], sizes)
      self.weights = []

   def train(self, dataset):
      NotImplemented

   def predict(self, example):
      NotImplemented

class NNUnit:
   """Unit of a neural net."""
   def __init__(self): 
       NotImplemented

class PerceptronLearner(NeuralNetLearner):

   def predict(self, example):
      return sum([])
#______________________________________________________________________________

class Linearlearner(Learner):
   """Fit a linear model to the data."""

   NotImplemented
#______________________________________________________________________________

class EnsembleLearner(Learner):
    """Given a list of learning algorithms, have them vote."""

    def __init__(self, learners=[]):
        self.learners=learners

    def train(self, dataset):
        for learner in self.learners:
           learner.train(dataset)

    def predict(self, example):
        return mode([learner.predict(example) for learner in self.learners])
        
#_____________________________________________________________________________
# Functions for testing learners on examples

def test(learner, dataset, examples=None, verbose=0):
    """Return the proportion of the examples that are correctly predicted.
    Assumes the learner has already been trained."""
    if examples == None: examples = dataset.examples
    if len(examples) == 0: return 0.0
    right = 0.0
    for example in examples:
        desired = example[dataset.target]
        output = learner.predict(dataset.sanitize(example))
        if output == desired:
            right += 1
            if verbose >= 2:
               print '   OK: got %s for %s' % (desired, example)
        elif verbose:
            print 'WRONG: got %s, expected %s for %s' % (
               output, desired, example)
    return right / len(examples)

def train_and_test(learner, dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder.
    Return the proportion of examples correct on the test examples."""
    examples = dataset.examples
    try:
        dataset.examples = examples[:start] + examples[end:]
        learner.dataset = dataset 
        learner.train(dataset)
        return test(learner, dataset, examples[start:end])
    finally:
        dataset.examples = examples

def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    if k == None:
        k = len(dataset.examples)
    if trials > 1:
        return mean([cross_validation(learner, dataset, k, trials=1)
                     for t in range(trials)])
    else:
        n = len(dataset.examples)
        random.shuffle(dataset.examples)
        return mean([train_and_test(learner, dataset, i*(n/k), (i+1)*(n/k))
                     for i in range(k)])
    
def leave1out(learner, dataset):
    "Leave one out cross-validation over the dataset."
    return cross_validation(learner, dataset, k=len(dataset.examples))

def learningcurve(learner, dataset, trials=10, sizes=None):
    if sizes == None:
        sizes = range(2, len(dataset.examples)-10, 2)
    def score(learner, size):
        random.shuffle(dataset.examples)
        return train_and_test(learner, dataset, 0, size)
    return [(size, mean([score(learner, size) for t in range(trials)]))
            for size in sizes]

#______________________________________________________________________________
# The rest of this file gives Data sets for machine learning problems.

orings = DataSet(name='orings', target='Distressed',
                 attrnames="Rings Distressed Temp Pressure Flightnum")


zoo = DataSet(name='zoo', target='type', exclude=['name'],
              attrnames="name hair feathers eggs milk airborne aquatic " +
              "predator toothed backbone breathes venomous fins legs tail " +
              "domestic catsize type") 


iris = DataSet(name="iris", target="class",
               attrnames="sepal-len sepal-width petal-len petal-width class")

#______________________________________________________________________________
# The Restaurant example from Fig. 18.2

def RestaurantDataSet(examples=None):
    "Build a DataSet of Restaurant waiting examples."
    return DataSet(name='restaurant', target='Wait', examples=examples,
                  attrnames='Alternate Bar Fri/Sat Hungry Patrons Price '
                   + 'Raining Reservation Type WaitEstimate Wait')

restaurant = RestaurantDataSet()

def T(attrname, branches):
    return DecisionTree(restaurant.attrnum(attrname), attrname, branches)

Fig[18,2] = T('Patrons',
             {'None': 'No', 'Some': 'Yes', 'Full':
              T('WaitEstimate',
                {'>60': 'No', '0-10': 'Yes', 
                 '30-60':
                 T('Alternate', {'No':
                                 T('Reservation', {'Yes': 'Yes', 'No':
                                                   T('Bar', {'No':'No',
                                                             'Yes':'Yes'})}),
                                 'Yes':
                                 T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                 '10-30':
                 T('Hungry', {'No': 'Yes', 'Yes':
                           T('Alternate',
                             {'No': 'Yes', 'Yes':
                              T('Raining', {'No': 'No', 'Yes': 'Yes'})})})})})

def SyntheticRestaurant(n=20):
    "Generate a DataSet with n examples."
    def gen():
        example =  map(random.choice, restaurant.values)
        example[restaurant.target] = Fig[18,2].predict(example)
        return example
    return RestaurantDataSet([gen() for i in range(n)])

#______________________________________________________________________________
# Artificial, generated  examples.

def Majority(k, n):
    """Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) > k/2)
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
    "2 inputs are chosen uniformly form (0.0 .. 2.0]; output is xor of ints."
    examples = []
    for i in range(n):
        x, y = [random.uniform(0.0, 2.0) for i in '12']
        examples.append([x, y, int(x) != int(y)])
    return DataSet(name="continuous xor", examples=examples)

#______________________________________________________________________________

def compare(algorithms=[MajorityLearner, NaiveBayesLearner, 
                        NearestNeighborLearner, DecisionTreeLearner],
            datasets=[iris, orings, zoo, restaurant, SyntheticRestaurant(20),
                      Majority(7, 100), Parity(7, 100), Xor(100)],
            k=10, trials=1):
    """Compare various learners on various datasets using cross-validation.
    Print results as a table."""
    print_table([[a.__name__.replace('Learner','')] +
                 [cross_validation(a(), d, k, trials) for d in datasets]
                 for a in algorithms],
                header=[''] + [d.name[0:7] for d in datasets], round=2)
    
