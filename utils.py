"""Provide some widely useful utilities. Safe for "from utils import *".

TODO: Let's take the >>> doctest examples out of the docstrings, and put them in utils_test.py
TODO: count_if and the like are leftovers from COmmon Lisp; let's make replace thenm with Pythonic alternatives.
TODO: if_ is a terrible idea; replace all uses with (x if test else y) and remove if_

TODO: Priority queues may not belong here -- see treatment in search.py
"""

import operator
import math
import random
import os.path
import bisect
import re

#______________________________________________________________________________
# Simple Data Structures: infinity, Dict, Struct

infinity = float('inf')

class Struct:
    """Create an instance with argument=value slots.
    
    This is for making a lightweight object whose class doesn't matter."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['{!s}={!s}'.format(k, repr(v)) 
                    for (k, v) in vars(self).items()]

def update(x, **entries):
    """Update a dict or an object with slots according to entries.
    
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)

    return x

#______________________________________________________________________________
# Functions on Sequences (mostly inspired by Common Lisp)
# NOTE: Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurences of item removed.
    >>> removeall(3, [1, 2, 3, 3, 2, 1, 3])
    [1, 2, 2, 1]
    >>> removeall(4, [1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements.
    >>> unique([1, 2, 3, 2, 1])
    [1, 2, 3]
    """
    return list(set(seq))

def product(numbers):
    """Return the product of the numbers.
    >>> product([1,2,3,4])
    24
    """
    return reduce(operator.mul, numbers, 1)

def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    return sum(map(lambda x: bool(predicate(x)), seq))

def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): 
            return x

    return None

def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """

    return all(predicate(x) for x in seq)

def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    elem = find_if(predicate,seq)

    return predicate(elem) or False

# TODO: rename to is_in or possibily add 'identity' to function name to clarify intent
def isin(elt, seq):
    """Like (elt in seq), but compares with is, not ==.
    >>> e = []; isin(e, [1, e, 3])
    True
    >>> isin(e, [1, [], 3])
    False
    """
    return any(x is elt for x in seq)

#______________________________________________________________________________
# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.

def argmin(seq, fn):
    return min(seq, key=fn)

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    smallest_score = min(seq, key=fn)

    return [elem for elem in seq if fn(elem) == smallest_score]

def argmin_gen(seq, fn):
    """Return a generator of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """

    smallest_score = min(seq, key=fn)

    yield from (elem for elem in seq if fn(elem) == smallest_score)

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    return random.choice(argmin_gen(seq, fn))

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_gen(seq, fn):
    """Return a generator of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    yield from argmin_gen(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))

#______________________________________________________________________________
# Statistical and mathematical functions

def histogram(values, mode=0, bin_function=None):
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first."""
    if bin_function: 
        values = map(bin_function, values)

    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1

    if mode:
        return sorted(bins.items(), key=lambda x: (x[1],x[0]), reverse=True)
    else:
        return sorted(bins.items())

from math import log2
from statistics import mode, median, mean, stdev

def stddev(values, meanval=None):
    """The standard deviation of a set of values.
    Pass in the mean if you already know it. """
    return stdev(values, mu=meanval)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def vector_add(a, b):
    """Component-wise addition of two vectors.
    >>> vector_add((0, 1), (8, 9))
    (8, 10)
    """
    return tuple(map(operator.add, a, b))

def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)

def weighted_sample_with_replacement(seq, weights, n):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]

def weighted_sampler(seq, weights):
    "Return a random-sample function that picks from seq weighted by weights."
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it.
    >>> num_or_str('42')
    42
    >>> num_or_str(' 42x ')
    '42x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def clip(x, lowest, highest):
    """Return x clipped to the range [lowest..highest].
    >>> [clip(x, 0, 1) for x in [-1, 0.5, 10]]
    [0, 0.5, 1]
    """
    return max(lowest, min(x, highest))


#______________________________________________________________________________
# Misc Functions

def printf(format_str, *args):
    """Format args with the first argument as format string, and write.
    Return the last arg, or format itself if there are no args."""
    print(str(format_str).format(*args, end=''))

    return args[-1] if args else format_str

def caller(n=1):
    """Return the name of the calling function n levels up in the frame stack.
    >>> caller(0)
    'caller'
    >>> def f():
    ...     return caller()
    >>> f()
    'f'
    """
    import inspect

    return inspect.getouterframes(inspect.currentframe())[n][3]

# TODO: Use functools.lru_cache memoization decorator
def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}

    return memoized_fn

def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if test:
        if callable(result): 
            return result()

        return result
    else:
        if callable(alternative): 
            return alternative()

        return alternative

def name(obj):
    "Try to find some reasonable name for the object."
    return (getattr(obj, 'name', 0) or getattr(obj, '__name__', 0)
            or getattr(getattr(obj, '__class__', 0), '__name__', 0)
            or str(obj))

def isnumber(x):
    "Is x a number? We say it is if it has a __int__ method."
    return hasattr(x, '__int__')

def issequence(x):
    "Is x a sequence? We say it is if it has a __getitem__ method."
    return hasattr(x, '__getitem__')

def print_table(table, header=None, sep='   ', numfmt='%g'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in different columns, don't use print_table.)
    sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
                for row in table]

    maxlen = lambda seq: max(map(len, seq))

    sizes = map(maxlen, zip(*[map(str, row) for row in table]))

    for row in table:
        print(sep.join(getattr(str(x), j)(size) 
                for (j, size, x) in zip(justs, sizes, row)))

def AIMAFile(components, mode='r'):
    "Open a file based at the AIMA root directory."
    import utils
    aima_root = os.path.dirname(utils.__file__)

    aima_file = os.path.join(aima_root, *components)

    return open(aima_file)

def DataFile(name, mode='r'):
    "Return a file in the AIMA /data directory."
    return AIMAFile(['..', 'data', name], mode)

def unimplemented():
    "Use this as a stub for not-yet-implemented functions."
    raise NotImplementedError

#______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue

# TODO: Use queue.Queue
class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""
    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items: self.append(item)

def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []

class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]

# TODO: Use queue.PriorityQueue
class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""
    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return some(lambda _, x: x == item, self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

## Fig: The idea is we can define things like Fig[3,10] later.
## Alas, it is Fig[3,10] not Fig[3.10], because that would be the same
## as Fig[3.1]
Fig = {}

#______________________________________________________________________________
# Support for doctest

def ignore(x):
    pass

def random_tests(text):
    """Some functions are stochastic. We want to be able to write a test
    with random output.  We do that by ignoring the output."""
    def fixup(test):
        return ">>> {}".format("ignore(" + test + ")" if " = " not in test else test)

    tests = re.findall(">>> (.*)", text)
    return '\n'.join(map(fixup, tests))
