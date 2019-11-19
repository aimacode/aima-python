"""
Representations and Inference for Logic. (Chapters 7-9, 12)

Covers both Propositional and First-Order Logic. First we have four
important data types:

    KB            Abstract class holds a knowledge base of logical expressions
    KB_Agent      Abstract class subclasses agents.Agent
    Expr          A logical expression, imported from utils.py
    substitution  Implemented as a dictionary of var:value pairs, {x:1, y:x}

Be careful: some functions take an Expr as argument, and some take a KB.

Logical expressions can be created with Expr or expr, imported from utils, TODO
or with expr, which adds the capability to write a string that uses
the connectives ==>, <==, <=>, or <=/=>. But be careful: these have the
operator precedence of commas; you may need to add parens to make precedence work.
See logic.ipynb for examples.

Then we implement various functions for doing logical inference:

    pl_true          Evaluate a propositional logical sentence in a model
    tt_entails       Say if a statement is entailed by a KB
    pl_resolution    Do resolution on propositional sentences
    dpll_satisfiable See if a propositional sentence is satisfiable
    WalkSAT          Try to find a solution for a set of clauses

And a few other functions:

    to_cnf           Convert to conjunctive normal form
    unify            Do unification of two FOL sentences
    diff, simp       Symbolic differentiation and simplification
"""

import heapq
import itertools
import random
from collections import defaultdict, Counter

import networkx as nx

from agents import Agent, Glitter, Bump, Stench, Breeze, Scream
from csp import parse_neighbors, UniversalDict
from search import astar_search, PlanRoute
from utils import remove_all, unique, first, probability, isnumber, issequence, Expr, expr, subexpressions, extend


class KB:
    """A knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract. Why ask_generator instead of ask?
    The book is a bit vague on what ask means --
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
    FOL KB, something like ask(Brother(x, y)) might return many substitutions
    such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        """Add the sentence to the KB."""
        raise NotImplementedError

    def ask(self, query):
        """Return a substitution that makes the query true, or, failing that, return False."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        """Yield all the substitutions that make query true."""
        raise NotImplementedError

    def retract(self, sentence):
        """Remove sentence from the KB."""
        raise NotImplementedError


class PropKB(KB):
    """A KB for propositional logic. Inefficient, with no indexing."""

    def __init__(self, sentence=None):
        super().__init__(sentence)
        self.clauses = []

    def tell(self, sentence):
        """Add the sentence's clauses to the KB."""
        self.clauses.extend(conjuncts(to_cnf(sentence)))

    def ask_generator(self, query):
        """Yield the empty substitution {} if KB entails query; else no results."""
        if tt_entails(Expr('&', *self.clauses), query):
            yield {}

    def ask_if_true(self, query):
        """Return True if the KB entails query, else return False."""
        for _ in self.ask_generator(query):
            return True
        return False

    def retract(self, sentence):
        """Remove the sentence's clauses from the KB."""
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)


# ______________________________________________________________________________


def KBAgentProgram(kb):
    """
    [Figure 7.1]
    A generic logical knowledge-based agent program.
    """
    steps = itertools.count()

    def program(percept):
        t = next(steps)
        kb.tell(make_percept_sentence(percept, t))
        action = kb.ask(make_action_query(t))
        kb.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(percept, t):
        return Expr('Percept')(percept, t)

    def make_action_query(t):
        return expr('ShouldDo(action, {})'.format(t))

    def make_action_sentence(action, t):
        return Expr('Did')(action[expr('action')], t)

    return program


def is_symbol(s):
    """A string s is a symbol if it starts with an alphabetic char.
    >>> is_symbol('R2D2')
    True
    """
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    """A logic variable symbol is an initial-lowercase string.
    >>> is_var_symbol('EXE')
    False
    """
    return is_symbol(s) and s[0].islower()


def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string.
    >>> is_prop_symbol('exe')
    False
    """
    return is_symbol(s) and s[0].isupper()


def variables(s):
    """Return a set of the variables in expression s.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) == {x, y, z}
    True
    """
    return {x for x in subexpressions(s) if is_variable(x)}


def is_definite_clause(s):
    """Returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive. In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
    else:
        return False


def parse_definite_clause(s):
    """Return the antecedents and the consequent of a definite clause."""
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent


# Useful constant Exprs used in examples and code:
A, B, C, D, E, F, G, P, Q, a, x, y, z, u = map(Expr, 'ABCDEFGPQaxyzu')


# ______________________________________________________________________________


def tt_entails(kb, alpha):
    """
    [Figure 7.10]
    Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. Note that the 'kb' should be an Expr which is a
    conjunction of clauses.
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    assert not variables(alpha)
    symbols = list(prop_symbols(kb & alpha))
    return tt_check_all(kb, alpha, symbols, {})


def tt_check_all(kb, alpha, symbols, model):
    """Auxiliary routine to implement tt_entails."""
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            return result
        else:
            return True
    else:
        P, rest = symbols[0], symbols[1:]
        return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
                tt_check_all(kb, alpha, rest, extend(model, P, False)))


def prop_symbols(x):
    """Return the set of all propositional symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op):
        return {x}
    else:
        return {symbol for arg in x.args for symbol in prop_symbols(arg)}


def constant_symbols(x):
    """Return the set of all constant symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op) and not x.args:
        return {x}
    else:
        return {symbol for arg in x.args for symbol in constant_symbols(arg)}


def predicate_symbols(x):
    """Return a set of (symbol_name, arity) in x.
    All symbols (even functional) with arity > 0 are considered."""
    if not isinstance(x, Expr) or not x.args:
        return set()
    pred_set = {(x.op, len(x.args))} if is_prop_symbol(x.op) else set()
    pred_set.update({symbol for arg in x.args for symbol in predicate_symbols(arg)})
    return pred_set


def tt_true(s):
    """Is a propositional sentence a tautology?
    >>> tt_true('P | ~P')
    True
    """
    s = expr(s)
    return tt_entails(True, s)


def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological.
    >>> pl_true(P, {}) is None
    True
    """
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError('Illegal operator in logic expression' + str(exp))


# ______________________________________________________________________________

# Convert to Conjunctive Normal Form (CNF)


def to_cnf(s):
    """
    [Page 253]
    Convert a propositional logical sentence to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...)
    >>> to_cnf('~(B | C)')
    (~B & ~C)
    """
    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)  # Steps 1, 2 from p. 253
    s = move_not_inwards(s)  # Step 3
    return distribute_and_over_or(s)  # Step 4


def eliminate_implications(s):
    """Change implications into equivalent form with only &, |, and ~ as logical operators."""
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s  # Atoms are unchanged.
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2  # TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    """
    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)

        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])  # ~~A ==> A
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    """Given an associative op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args).
    >>> dissociate('&', [A & B])
    [A, B]
    """
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])


def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])


# ______________________________________________________________________________


def pl_resolution(kb, alpha):
    """
    [Figure 7.12]
    Propositional-logic resolution: say if alpha follows from KB.
    >>> pl_resolution(horn_clauses_KB, A)
    True
    """
    clauses = kb.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i + 1, n)]
        for (ci, cj) in pairs:
            resolvents = pl_resolve(ci, cj)
            if False in resolvents:
                return True
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)):
            return False
        for c in new:
            if c not in clauses:
                clauses.append(c)


def pl_resolve(ci, cj):
    """Return all clauses that can be obtained by resolving clauses ci and cj."""
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                clauses.append(associate('|', unique(remove_all(di, disjuncts(ci)) + remove_all(dj, disjuncts(cj)))))
    return clauses


# ______________________________________________________________________________


class PropDefiniteKB(PropKB):
    """A KB of propositional definite clauses."""

    def tell(self, sentence):
        """Add a definite clause to this KB."""
        assert is_definite_clause(sentence), "Must be definite clause"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        """Yield the empty substitution if KB implies query; else nothing."""
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses if c.op == '==>' and p in conjuncts(c.args[0])]


def pl_fc_entails(kb, q):
    """
    [Figure 7.15]
    Use forward chaining to see if a PropDefiniteKB entails symbol q.
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """
    count = {c: len(conjuncts(c.args[0])) for c in kb.clauses if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in kb.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in kb.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False


"""
[Figure 7.13]
Simple inference in a wumpus world example
"""
wumpus_world_inference = expr('(B11 <=> (P12 | P21))  &  ~B11')

"""
[Figure 7.16]
Propositional Logic Forward Chaining example
"""
horn_clauses_KB = PropDefiniteKB()
for clause in ['P ==> Q',
               '(L & M) ==> P',
               '(B & L) ==> M',
               '(A & P) ==> L',
               '(A & B) ==> L',
               'A', 'B']:
    horn_clauses_KB.tell(expr(clause))

"""
Definite clauses KB example
"""
definite_clauses_KB = PropDefiniteKB()
for clause in ['(B & F) ==> E',
               '(A & E & F) ==> G',
               '(B & C) ==> F',
               '(A & B) ==> D',
               '(E & F) ==> H',
               '(H & I) ==>J',
               'A', 'B', 'C']:
    definite_clauses_KB.tell(expr(clause))


# ______________________________________________________________________________
# Heuristics for SAT Solvers


def no_branching_heuristic(symbols, clauses):
    return first(symbols), True


def min_clauses(clauses):
    min_len = min(map(lambda c: len(c.args), clauses), default=2)
    return filter(lambda c: len(c.args) == (min_len if min_len > 1 else 2), clauses)


def moms(symbols, clauses):
    """
    MOMS (Maximum Occurrence in clauses of Minimum Size) heuristic
    Returns the literal with the most occurrences in all clauses of minimum size
    """
    scores = Counter(l for c in min_clauses(clauses) for l in prop_symbols(c))
    return max(symbols, key=lambda symbol: scores[symbol]), True


def momsf(symbols, clauses, k=0):
    """
    MOMS alternative heuristic
    If f(x) the number of occurrences of the variable x in clauses with minimum size,
    we choose the variable maximizing [f(x) + f(-x)] * 2^k + f(x) * f(-x)
    Returns x if f(x) >= f(-x) otherwise -x
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c))
    P = max(symbols,
            key=lambda symbol: (scores[symbol] + scores[~symbol]) * pow(2, k) + scores[symbol] * scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def posit(symbols, clauses):
    """
    Freeman's POSIT version of MOMs
    Counts the positive x and negative x for each variable x in clauses with minimum size
    Returns x if f(x) >= f(-x) otherwise -x
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def zm(symbols, clauses):
    """
    Zabih and McAllester's version of MOMs
    Counts the negative occurrences only of each variable x in clauses with minimum size
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c) if l.op == '~')
    return max(symbols, key=lambda symbol: scores[~symbol]), True


def dlis(symbols, clauses):
    """
    DLIS (Dynamic Largest Individual Sum) heuristic
    Choose the variable and value that satisfies the maximum number of unsatisfied clauses
    Like DLCS but we only consider the literal (thus Cp and Cn are individual)
    """
    scores = Counter(l for c in clauses for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol])
    return P, True if scores[P] >= scores[~P] else False


def dlcs(symbols, clauses):
    """
    DLCS (Dynamic Largest Combined Sum) heuristic
    Cp the number of clauses containing literal x
    Cn the number of clauses containing literal -x
    Here we select the variable maximizing Cp + Cn
    Returns x if Cp >= Cn otherwise -x
    """
    scores = Counter(l for c in clauses for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def jw(symbols, clauses):
    """
    Jeroslow-Wang heuristic
    For each literal compute J(l) = \sum{l in clause c} 2^{-|c|}
    Return the literal maximizing J
    """
    scores = Counter()
    for c in clauses:
        for l in prop_symbols(c):
            scores[l] += pow(2, -len(c.args))
    return max(symbols, key=lambda symbol: scores[symbol]), True


def jw2(symbols, clauses):
    """
    Two Sided Jeroslow-Wang heuristic
    Compute J(l) also counts the negation of l = J(x) + J(-x)
    Returns x if J(x) >= J(-x) otherwise -x
    """
    scores = Counter()
    for c in clauses:
        for l in disjuncts(c):
            scores[l] += pow(2, -len(c.args))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


# ______________________________________________________________________________
# DPLL-Satisfiable [Figure 7.17]


def dpll_satisfiable(s, branching_heuristic=no_branching_heuristic):
    """Check satisfiability of a propositional sentence.
    This differs from the book code in two ways: (1) it returns a model
    rather than True when it succeeds; this is more useful. (2) The
    function find_pure_symbol is passed a list of unknown clauses, rather
    than a list of all clauses and the model; this is more efficient.
    >>> dpll_satisfiable(A |'<=>'| B) == {A: True, B: True}
    True
    """
    return dpll(conjuncts(to_cnf(s)), prop_symbols(s), {}, branching_heuristic)


def dpll(clauses, symbols, model, branching_heuristic=no_branching_heuristic):
    """See if the clauses are true in a partial model."""
    unknown_clauses = []  # clauses with an unknown truth value
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is None:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = branching_heuristic(symbols, unknown_clauses)
    return (dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic) or
            dpll(clauses, remove_all(P, symbols), extend(model, P, not value), branching_heuristic))


def find_pure_symbol(symbols, clauses):
    """Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.
    >>> find_pure_symbol([A, B, C], [A|~B,~B|~C,C|A])
    (A, True)
    """
    for s in symbols:
        found_pos, found_neg = False, False
        for c in clauses:
            if not found_pos and s in disjuncts(c):
                found_pos = True
            if not found_neg and ~s in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return s, found_pos
    return None, None


def find_unit_clause(clauses, model):
    """Find a forced assignment if possible from a clause with only 1
    variable not bound in the model.
    >>> find_unit_clause([A|B|C, B|~C, ~A|~B], {A:True})
    (B, False)
    """
    for clause in clauses:
        P, value = unit_clause_assign(clause, model)
        if P:
            return P, value
    return None, None


def unit_clause_assign(clause, model):
    """Return a single variable/value pair that makes clause true in
    the model, if possible.
    >>> unit_clause_assign(A|B|C, {A:True})
    (None, None)
    >>> unit_clause_assign(B|~C, {A:True})
    (None, None)
    >>> unit_clause_assign(~A|~B, {A:True})
    (B, False)
    """
    P, value = None, None
    for literal in disjuncts(clause):
        sym, positive = inspect_literal(literal)
        if sym in model:
            if model[sym] == positive:
                return None, None  # clause already True
        elif P:
            return None, None  # more than 1 unbound variable
        else:
            P, value = sym, positive
    return P, value


def inspect_literal(literal):
    """The symbol in this literal, and the value it should take to
    make the literal true.
    >>> inspect_literal(P)
    (P, True)
    >>> inspect_literal(~P)
    (P, False)
    """
    if literal.op == '~':
        return literal.args[0], False
    else:
        return literal, True


# ______________________________________________________________________________
# CDCL - Conflict-Driven Clause Learning with 1UIP Learning Scheme,
# 2WL Lazy Data Structure, VSIDS Branching Heuristic & Restarts


def no_restart(conflicts, restarts, queue_lbd, sum_lbd):
    return False


def luby(conflicts, restarts, queue_lbd, sum_lbd, unit=512):
    # in the state-of-art tested with unit value 1, 2, 4, 6, 8, 12, 16, 32, 64, 128, 256 and 512
    def _luby(i):
        k = 1
        while True:
            if i == (1 << k) - 1:
                return 1 << (k - 1)
            elif (1 << (k - 1)) <= i < (1 << k) - 1:
                return _luby(i - (1 << (k - 1)) + 1)
            k += 1

    return unit * _luby(restarts) == len(queue_lbd)


def glucose(conflicts, restarts, queue_lbd, sum_lbd, x=100, k=0.7):
    # in the state-of-art tested with (x, k) as (50, 0.8) and (100, 0.7)
    # if there were at least x conflicts since the last restart, and then the average LBD of the last
    # x learnt clauses was at least k times higher than the average LBD of all learnt clauses
    return len(queue_lbd) >= x and sum(queue_lbd) / len(queue_lbd) * k > sum_lbd / conflicts


def cdcl_satisfiable(s, vsids_decay=0.95, restart_strategy=no_restart):
    """
    >>> cdcl_satisfiable(A |'<=>'| B) == {A: True, B: True}
    True
    """
    clauses = TwoWLClauseDatabase(conjuncts(to_cnf(s)))
    symbols = prop_symbols(s)
    scores = Counter()
    G = nx.DiGraph()
    model = {}
    dl = 0
    conflicts = 0
    restarts = 1
    sum_lbd = 0
    queue_lbd = []
    while True:
        conflict = unit_propagation(clauses, symbols, model, G, dl)
        if conflict:
            if dl == 0:
                return False
            conflicts += 1
            dl, learn, lbd = conflict_analysis(G, dl)
            queue_lbd.append(lbd)
            sum_lbd += lbd
            backjump(symbols, model, G, dl)
            clauses.add(learn, model)
            scores.update(l for l in disjuncts(learn))
            for symbol in scores:
                scores[symbol] *= vsids_decay
            if restart_strategy(conflicts, restarts, queue_lbd, sum_lbd):
                backjump(symbols, model, G)
                queue_lbd.clear()
                restarts += 1
        else:
            if not symbols:
                return model
            dl += 1
            assign_decision_literal(symbols, model, scores, G, dl)


def assign_decision_literal(symbols, model, scores, G, dl):
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    value = True if scores[P] >= scores[~P] else False
    symbols.remove(P)
    model[P] = value
    G.add_node(P, val=value, dl=dl)


def unit_propagation(clauses, symbols, model, G, dl):
    def check(c):
        if not model or clauses.get_first_watched(c) == clauses.get_second_watched(c):
            return True
        w1, _ = inspect_literal(clauses.get_first_watched(c))
        if w1 in model:
            return c in (clauses.get_neg_watched(w1) if model[w1] else clauses.get_pos_watched(w1))
        w2, _ = inspect_literal(clauses.get_second_watched(c))
        if w2 in model:
            return c in (clauses.get_neg_watched(w2) if model[w2] else clauses.get_pos_watched(w2))

    def unit_clause(watching):
        w, p = inspect_literal(watching)
        G.add_node(w, val=p, dl=dl)
        G.add_edges_from(zip(prop_symbols(c) - {w}, itertools.cycle([w])), antecedent=c)
        symbols.remove(w)
        model[w] = p

    def conflict_clause(c):
        G.add_edges_from(zip(prop_symbols(c), itertools.cycle('K')), antecedent=c)

    while True:
        bcp = False
        for c in filter(check, clauses.get_clauses()):
            # we need only visit each clause when one of its two watched literals is assigned to 0 because, until
            # this happens, we can guarantee that there cannot be more than n-2 literals in the clause assigned to 0
            first_watched = pl_true(clauses.get_first_watched(c), model)
            second_watched = pl_true(clauses.get_second_watched(c), model)
            if first_watched is None and clauses.get_first_watched(c) == clauses.get_second_watched(c):
                unit_clause(clauses.get_first_watched(c))
                bcp = True
                break
            elif first_watched is False and second_watched is not True:
                if clauses.update_second_watched(c, model):
                    bcp = True
                else:
                    # if the only literal with a non-zero value is the other watched literal then
                    if second_watched is None:  # if it is free, then the clause is a unit clause
                        unit_clause(clauses.get_second_watched(c))
                        bcp = True
                        break
                    else:  # else (it is False) the clause is a conflict clause
                        conflict_clause(c)
                        return True
            elif second_watched is False and first_watched is not True:
                if clauses.update_first_watched(c, model):
                    bcp = True
                else:
                    # if the only literal with a non-zero value is the other watched literal then
                    if first_watched is None:  # if it is free, then the clause is a unit clause
                        unit_clause(clauses.get_first_watched(c))
                        bcp = True
                        break
                    else:  # else (it is False) the clause is a conflict clause
                        conflict_clause(c)
                        return True
        if not bcp:
            return False


def conflict_analysis(G, dl):
    conflict_clause = next(G[p]['K']['antecedent'] for p in G.pred['K'])
    P = next(node for node in G.nodes() - 'K' if G.nodes[node]['dl'] == dl and G.in_degree(node) == 0)
    first_uip = nx.immediate_dominators(G, P)['K']
    G.remove_node('K')
    conflict_side = nx.descendants(G, first_uip)
    while True:
        for l in prop_symbols(conflict_clause).intersection(conflict_side):
            antecedent = next(G[p][l]['antecedent'] for p in G.pred[l])
            conflict_clause = pl_binary_resolution(conflict_clause, antecedent)
            # the literal block distance is calculated by taking the decision levels from variables of all
            # literals in the clause, and counting how many different decision levels were in this set
            lbd = [G.nodes[l]['dl'] for l in prop_symbols(conflict_clause)]
            if lbd.count(dl) == 1 and first_uip in prop_symbols(conflict_clause):
                return 0 if len(lbd) == 1 else heapq.nlargest(2, lbd)[-1], conflict_clause, len(set(lbd))


def pl_binary_resolution(ci, cj):
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                return pl_binary_resolution(associate('|', remove_all(di, disjuncts(ci))),
                                            associate('|', remove_all(dj, disjuncts(cj))))
    return associate('|', unique(disjuncts(ci) + disjuncts(cj)))


def backjump(symbols, model, G, dl=0):
    delete = {node for node in G.nodes() if G.nodes[node]['dl'] > dl}
    G.remove_nodes_from(delete)
    for node in delete:
        del model[node]
    symbols |= delete


class TwoWLClauseDatabase:

    def __init__(self, clauses):
        self.__twl = {}
        self.__watch_list = defaultdict(lambda: [set(), set()])
        for c in clauses:
            self.add(c, None)

    def get_clauses(self):
        return self.__twl.keys()

    def set_first_watched(self, clause, new_watching):
        if len(clause.args) > 2:
            self.__twl[clause][0] = new_watching

    def set_second_watched(self, clause, new_watching):
        if len(clause.args) > 2:
            self.__twl[clause][1] = new_watching

    def get_first_watched(self, clause):
        if len(clause.args) == 2:
            return clause.args[0]
        if len(clause.args) > 2:
            return self.__twl[clause][0]
        return clause

    def get_second_watched(self, clause):
        if len(clause.args) == 2:
            return clause.args[-1]
        if len(clause.args) > 2:
            return self.__twl[clause][1]
        return clause

    def get_pos_watched(self, l):
        return self.__watch_list[l][0]

    def get_neg_watched(self, l):
        return self.__watch_list[l][1]

    def add(self, clause, model):
        self.__twl[clause] = self.__assign_watching_literals(clause, model)
        w1, p1 = inspect_literal(self.get_first_watched(clause))
        w2, p2 = inspect_literal(self.get_second_watched(clause))
        self.__watch_list[w1][0].add(clause) if p1 else self.__watch_list[w1][1].add(clause)
        if w1 != w2:
            self.__watch_list[w2][0].add(clause) if p2 else self.__watch_list[w2][1].add(clause)

    def remove(self, clause):
        w1, p1 = inspect_literal(self.get_first_watched(clause))
        w2, p2 = inspect_literal(self.get_second_watched(clause))
        del self.__twl[clause]
        self.__watch_list[w1][0].discard(clause) if p1 else self.__watch_list[w1][1].discard(clause)
        if w1 != w2:
            self.__watch_list[w2][0].discard(clause) if p2 else self.__watch_list[w2][1].discard(clause)

    def update_first_watched(self, clause, model):
        # if a non-zero literal different from the other watched literal is found
        found, new_watching = self.__find_new_watching_literal(clause, self.get_first_watched(clause), model)
        if found:  # then it will replace the watched literal
            w, p = inspect_literal(self.get_second_watched(clause))
            self.__watch_list[w][0].remove(clause) if p else self.__watch_list[w][1].remove(clause)
            self.set_second_watched(clause, new_watching)
            w, p = inspect_literal(new_watching)
            self.__watch_list[w][0].add(clause) if p else self.__watch_list[w][1].add(clause)
            return True

    def update_second_watched(self, clause, model):
        # if a non-zero literal different from the other watched literal is found
        found, new_watching = self.__find_new_watching_literal(clause, self.get_second_watched(clause), model)
        if found:  # then it will replace the watched literal
            w, p = inspect_literal(self.get_first_watched(clause))
            self.__watch_list[w][0].remove(clause) if p else self.__watch_list[w][1].remove(clause)
            self.set_first_watched(clause, new_watching)
            w, p = inspect_literal(new_watching)
            self.__watch_list[w][0].add(clause) if p else self.__watch_list[w][1].add(clause)
            return True

    def __find_new_watching_literal(self, clause, other_watched, model):
        # if a non-zero literal different from the other watched literal is found
        if len(clause.args) > 2:
            for l in disjuncts(clause):
                if l != other_watched and pl_true(l, model) is not False:
                    # then it is returned
                    return True, l
        return False, None

    def __assign_watching_literals(self, clause, model=None):
        if len(clause.args) > 2:
            if model is None or not model:
                return [clause.args[0], clause.args[-1]]
            else:
                return [next(l for l in disjuncts(clause) if pl_true(l, model) is None),
                        next(l for l in disjuncts(clause) if pl_true(l, model) is False)]


# ______________________________________________________________________________
# Walk-SAT [Figure 7.18]


def WalkSAT(clauses, p=0.5, max_flips=10000):
    """Checks for satisfiability of all clauses by randomly flipping values of variables
    >>> WalkSAT([A & ~A], 0.5, 100) is None
    True
    """
    # Set of all symbols in all clauses
    symbols = {sym for clause in clauses for sym in prop_symbols(clause)}
    # model is a random assignment of true/false to the symbols in clauses
    model = {s: random.choice([True, False]) for s in symbols}
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            (satisfied if pl_true(clause, model) else unsatisfied).append(clause)
        if not unsatisfied:  # if model satisfies all the clauses
            return model
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(list(prop_symbols(clause)))
        else:
            # Flip the symbol in clause that maximizes number of sat. clauses
            def sat_count(sym):
                # Return the the number of clauses satisfied after flipping the symbol.
                model[sym] = not model[sym]
                count = len([clause for clause in clauses if pl_true(clause, model)])
                model[sym] = not model[sym]
                return count

            sym = max(prop_symbols(clause), key=sat_count)
        model[sym] = not model[sym]
    # If no solution is found within the flip limit, we return failure
    return None


# ______________________________________________________________________________
# Map Coloring SAT Problems


def MapColoringSAT(colors, neighbors):
    """Make a SAT for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    colors = UniversalDict(colors)
    clauses = []
    for state in neighbors.keys():
        clause = [expr(state + '_' + c) for c in colors[state]]
        clauses.append(clause)
        for t in itertools.combinations(clause, 2):
            clauses.append([~t[0], ~t[1]])
        visited = set()
        adj = set(neighbors[state]) - visited
        visited.add(state)
        for n_state in adj:
            for col in colors[n_state]:
                clauses.append([expr('~' + state + '_' + col), expr('~' + n_state + '_' + col)])
    return associate('&', map(lambda c: associate('|', c), clauses))


australia_sat = MapColoringSAT(list('RGB'), """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """)

france_sat = MapColoringSAT(list('RGBY'),
                            """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""")

usa_sat = MapColoringSAT(list('RGBY'),
                         """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """)


# ______________________________________________________________________________


# Expr functions for WumpusKB and HybridWumpusAgent

def facing_east(time):
    return Expr('FacingEast', time)


def facing_west(time):
    return Expr('FacingWest', time)


def facing_north(time):
    return Expr('FacingNorth', time)


def facing_south(time):
    return Expr('FacingSouth', time)


def wumpus(x, y):
    return Expr('W', x, y)


def pit(x, y):
    return Expr('P', x, y)


def breeze(x, y):
    return Expr('B', x, y)


def stench(x, y):
    return Expr('S', x, y)


def wumpus_alive(time):
    return Expr('WumpusAlive', time)


def have_arrow(time):
    return Expr('HaveArrow', time)


def percept_stench(time):
    return Expr('Stench', time)


def percept_breeze(time):
    return Expr('Breeze', time)


def percept_glitter(time):
    return Expr('Glitter', time)


def percept_bump(time):
    return Expr('Bump', time)


def percept_scream(time):
    return Expr('Scream', time)


def move_forward(time):
    return Expr('Forward', time)


def shoot(time):
    return Expr('Shoot', time)


def turn_left(time):
    return Expr('TurnLeft', time)


def turn_right(time):
    return Expr('TurnRight', time)


def ok_to_move(x, y, time):
    return Expr('OK', x, y, time)


def location(x, y, time=None):
    if time is None:
        return Expr('L', x, y)
    else:
        return Expr('L', x, y, time)


# Symbols

def implies(lhs, rhs):
    return Expr('==>', lhs, rhs)


def equiv(lhs, rhs):
    return Expr('<=>', lhs, rhs)


# Helper Function

def new_disjunction(sentences):
    t = sentences[0]
    for i in range(1, len(sentences)):
        t |= sentences[i]
    return t


# ______________________________________________________________________________


class WumpusKB(PropKB):
    """
    Create a Knowledge Base that contains the a temporal "Wumpus physics" and temporal rules with time zero.
    """

    def __init__(self, dimrow):
        super().__init__()
        self.dimrow = dimrow
        self.tell(~wumpus(1, 1))
        self.tell(~pit(1, 1))

        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):

                pits_in = list()
                wumpus_in = list()

                if x > 1:  # West room exists
                    pits_in.append(pit(x - 1, y))
                    wumpus_in.append(wumpus(x - 1, y))

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell(equiv(stench(x, y), new_disjunction(wumpus_in)))

        # Rule that describes existence of at least one Wumpus
        wumpus_at_least = list()
        for x in range(1, dimrow + 1):
            for y in range(1, dimrow + 1):
                wumpus_at_least.append(wumpus(x, y))

        self.tell(new_disjunction(wumpus_at_least))

        # Rule that describes existence of at most one Wumpus
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                for u in range(1, dimrow + 1):
                    for v in range(1, dimrow + 1):
                        if i != u or j != v:
                            self.tell(~wumpus(i, j) | ~wumpus(u, v))

        # Temporal rules at time zero
        self.tell(location(1, 1, 0))
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                self.tell(implies(location(i, j, 0), equiv(percept_breeze(0), breeze(i, j))))
                self.tell(implies(location(i, j, 0), equiv(percept_stench(0), stench(i, j))))
                if i != 1 or j != 1:
                    self.tell(~location(i, j, 0))

        self.tell(wumpus_alive(0))
        self.tell(have_arrow(0))
        self.tell(facing_east(0))
        self.tell(~facing_north(0))
        self.tell(~facing_south(0))
        self.tell(~facing_west(0))

    def make_action_sentence(self, action, time):
        actions = [move_forward(time), shoot(time), turn_left(time), turn_right(time)]

        for a in actions:
            if action is a:
                self.tell(action)
            else:
                self.tell(~a)

    def make_percept_sentence(self, percept, time):
        # Glitter, Bump, Stench, Breeze, Scream
        flags = [0, 0, 0, 0, 0]

        # Things perceived
        if isinstance(percept, Glitter):
            flags[0] = 1
            self.tell(percept_glitter(time))
        elif isinstance(percept, Bump):
            flags[1] = 1
            self.tell(percept_bump(time))
        elif isinstance(percept, Stench):
            flags[2] = 1
            self.tell(percept_stench(time))
        elif isinstance(percept, Breeze):
            flags[3] = 1
            self.tell(percept_breeze(time))
        elif isinstance(percept, Scream):
            flags[4] = 1
            self.tell(percept_scream(time))

        # Things not perceived
        for i in range(len(flags)):
            if flags[i] == 0:
                if i == 0:
                    self.tell(~percept_glitter(time))
                elif i == 1:
                    self.tell(~percept_bump(time))
                elif i == 2:
                    self.tell(~percept_stench(time))
                elif i == 3:
                    self.tell(~percept_breeze(time))
                elif i == 4:
                    self.tell(~percept_scream(time))

    def add_temporal_sentences(self, time):
        if time == 0:
            return
        t = time - 1

        # current location rules
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                self.tell(implies(location(i, j, time), equiv(percept_breeze(time), breeze(i, j))))
                self.tell(implies(location(i, j, time), equiv(percept_stench(time), stench(i, j))))
                s = list()
                s.append(equiv(location(i, j, time), location(i, j, time) & ~move_forward(time) | percept_bump(time)))
                if i != 1:
                    s.append(location(i - 1, j, t) & facing_east(t) & move_forward(t))
                if i != self.dimrow:
                    s.append(location(i + 1, j, t) & facing_west(t) & move_forward(t))
                if j != 1:
                    s.append(location(i, j - 1, t) & facing_north(t) & move_forward(t))
                if j != self.dimrow:
                    s.append(location(i, j + 1, t) & facing_south(t) & move_forward(t))

                # add sentence about location i,j
                self.tell(new_disjunction(s))

                # add sentence about safety of location i,j
                self.tell(equiv(ok_to_move(i, j, time), ~pit(i, j) & ~wumpus(i, j) & wumpus_alive(time)))

        # Rules about current orientation

        a = facing_north(t) & turn_right(t)
        b = facing_south(t) & turn_left(t)
        c = facing_east(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_east(time), a | b | c)
        self.tell(s)

        a = facing_north(t) & turn_left(t)
        b = facing_south(t) & turn_right(t)
        c = facing_west(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_west(time), a | b | c)
        self.tell(s)

        a = facing_east(t) & turn_left(t)
        b = facing_west(t) & turn_right(t)
        c = facing_north(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_north(time), a | b | c)
        self.tell(s)

        a = facing_west(t) & turn_left(t)
        b = facing_east(t) & turn_right(t)
        c = facing_south(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_south(time), a | b | c)
        self.tell(s)

        # Rules about last action
        self.tell(equiv(move_forward(t), ~turn_right(t) & ~turn_left(t)))

        # Rule about the arrow
        self.tell(equiv(have_arrow(time), have_arrow(t) & ~shoot(t)))

        # Rule about Wumpus (dead or alive)
        self.tell(equiv(wumpus_alive(time), wumpus_alive(t) & ~percept_scream(time)))

    def ask_if_true(self, query):
        return pl_resolution(self, query)


# ______________________________________________________________________________


class WumpusPosition:
    def __init__(self, x, y, orientation):
        self.X = x
        self.Y = y
        self.orientation = orientation

    def get_location(self):
        return self.X, self.Y

    def set_location(self, x, y):
        self.X = x
        self.Y = y

    def get_orientation(self):
        return self.orientation

    def set_orientation(self, orientation):
        self.orientation = orientation

    def __eq__(self, other):
        if other.get_location() == self.get_location() and other.get_orientation() == self.get_orientation():
            return True
        else:
            return False


# ______________________________________________________________________________


class HybridWumpusAgent(Agent):
    """
    [Figure 7.20]
    An agent for the wumpus world that does logical inference.
    """

    def __init__(self, dimentions):
        self.dimrow = dimentions
        self.kb = WumpusKB(self.dimrow)
        self.t = 0
        self.plan = list()
        self.current_position = WumpusPosition(1, 1, 'UP')
        super().__init__(self.execute)

    def execute(self, percept):
        self.kb.make_percept_sentence(percept, self.t)
        self.kb.add_temporal_sentences(self.t)

        temp = list()

        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(location(i, j, self.t)):
                    temp.append(i)
                    temp.append(j)

        if self.kb.ask_if_true(facing_north(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'UP')
        elif self.kb.ask_if_true(facing_south(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'DOWN')
        elif self.kb.ask_if_true(facing_west(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'LEFT')
        elif self.kb.ask_if_true(facing_east(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'RIGHT')

        safe_points = list()
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(ok_to_move(i, j, self.t)):
                    safe_points.append([i, j])

        if self.kb.ask_if_true(percept_glitter(self.t)):
            goals = list()
            goals.append([1, 1])
            self.plan.append('Grab')
            actions = self.plan_route(self.current_position, goals, safe_points)
            self.plan.extend(actions)
            self.plan.append('Climb')

        if len(self.plan) == 0:
            unvisited = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    for k in range(self.t):
                        if self.kb.ask_if_true(location(i, j, k)):
                            unvisited.append([i, j])
            unvisited_and_safe = list()
            for u in unvisited:
                for s in safe_points:
                    if u not in unvisited_and_safe and s == u:
                        unvisited_and_safe.append(u)

            temp = self.plan_route(self.current_position, unvisited_and_safe, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0 and self.kb.ask_if_true(have_arrow(self.t)):
            possible_wumpus = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if not self.kb.ask_if_true(wumpus(i, j)):
                        possible_wumpus.append([i, j])

            temp = self.plan_shot(self.current_position, possible_wumpus, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0:
            not_unsafe = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if not self.kb.ask_if_true(ok_to_move(i, j, self.t)):
                        not_unsafe.append([i, j])
            temp = self.plan_route(self.current_position, not_unsafe, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0:
            start = list()
            start.append([1, 1])
            temp = self.plan_route(self.current_position, start, safe_points)
            self.plan.extend(temp)
            self.plan.append('Climb')

        action = self.plan[0]
        self.plan = self.plan[1:]
        self.kb.make_action_sentence(action, self.t)
        self.t += 1

        return action

    def plan_route(self, current, goals, allowed):
        problem = PlanRoute(current, goals, allowed, self.dimrow)
        return astar_search(problem).solution()

    def plan_shot(self, current, goals, allowed):
        shooting_positions = set()

        for loc in goals:
            x = loc[0]
            y = loc[1]
            for i in range(1, self.dimrow + 1):
                if i < x:
                    shooting_positions.add(WumpusPosition(i, y, 'EAST'))
                if i > x:
                    shooting_positions.add(WumpusPosition(i, y, 'WEST'))
                if i < y:
                    shooting_positions.add(WumpusPosition(x, i, 'NORTH'))
                if i > y:
                    shooting_positions.add(WumpusPosition(x, i, 'SOUTH'))

        # Can't have a shooting position from any of the rooms the Wumpus could reside
        orientations = ['EAST', 'WEST', 'NORTH', 'SOUTH']
        for loc in goals:
            for orientation in orientations:
                shooting_positions.remove(WumpusPosition(loc[0], loc[1], orientation))

        actions = list()
        actions.extend(self.plan_route(current, shooting_positions, allowed))
        actions.append('Shoot')
        return actions


# ______________________________________________________________________________


def SAT_plan(init, transition, goal, t_max, SAT_solver=cdcl_satisfiable):
    """
    [Figure 7.22]
    Converts a planning problem to Satisfaction problem by translating it to a cnf sentence.
    >>> transition = {'A': {'Left': 'A', 'Right': 'B'}, 'B': {'Left': 'A', 'Right': 'C'}, 'C': {'Left': 'B', 'Right': 'C'}}
    >>> SAT_plan('A', transition, 'C', 1) is None
    True
    """

    # Functions used by SAT_plan
    def translate_to_SAT(init, transition, goal, time):
        clauses = []
        states = [state for state in transition]

        # Symbol claiming state s at time t
        state_counter = itertools.count()
        for s in states:
            for t in range(time + 1):
                state_sym[s, t] = Expr('S_{}'.format(next(state_counter)))

        # Add initial state axiom
        clauses.append(state_sym[init, 0])

        # Add goal state axiom
        clauses.append(state_sym[first(clause[0] for clause in state_sym
                                       if set(conjuncts(clause[0])).issuperset(conjuncts(goal))), time]) \
            if isinstance(goal, Expr) else clauses.append(state_sym[goal, time])

        # All possible transitions
        transition_counter = itertools.count()
        for s in states:
            for action in transition[s]:
                s_ = transition[s][action]
                for t in range(time):
                    # Action 'action' taken from state 's' at time 't' to reach 's_'
                    action_sym[s, action, t] = Expr('T_{}'.format(next(transition_counter)))

                    # Change the state from s to s_
                    clauses.append(action_sym[s, action, t] | '==>' | state_sym[s, t])
                    clauses.append(action_sym[s, action, t] | '==>' | state_sym[s_, t + 1])

        # Allow only one state at any time
        for t in range(time + 1):
            # must be a state at any time
            clauses.append(associate('|', [state_sym[s, t] for s in states]))

            for s in states:
                for s_ in states[states.index(s) + 1:]:
                    # for each pair of states s, s_ only one is possible at time t
                    clauses.append((~state_sym[s, t]) | (~state_sym[s_, t]))

        # Restrict to one transition per timestep
        for t in range(time):
            # list of possible transitions at time t
            transitions_t = [tr for tr in action_sym if tr[2] == t]

            # make sure at least one of the transitions happens
            clauses.append(associate('|', [action_sym[tr] for tr in transitions_t]))

            for tr in transitions_t:
                for tr_ in transitions_t[transitions_t.index(tr) + 1:]:
                    # there cannot be two transitions tr and tr_ at time t
                    clauses.append(~action_sym[tr] | ~action_sym[tr_])

        # Combine the clauses to form the cnf
        return associate('&', clauses)

    def extract_solution(model):
        true_transitions = [t for t in action_sym if model[action_sym[t]]]
        # Sort transitions based on time, which is the 3rd element of the tuple
        true_transitions.sort(key=lambda x: x[2])
        return [action for s, action, time in true_transitions]

    # Body of SAT_plan algorithm
    for t in range(t_max + 1):
        # dictionaries to help extract the solution from model
        state_sym = {}
        action_sym = {}

        cnf = translate_to_SAT(init, transition, goal, t)
        model = SAT_solver(cnf)
        if model is not False:
            return extract_solution(model)
    return None


# ______________________________________________________________________________


def unify(x, y, s={}):
    """
    [Figure 9.1]
    Unify expressions x,y with substitution s; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. Expr('x')), constants, lists, or Exprs.
    >>> unify(x, 3, {})
    {x: 3}
    """
    if s is None:
        return None
    elif x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif isinstance(x, Expr) and isinstance(y, Expr):
        return unify(x.args, y.args, unify(x.op, y.op, s))
    elif isinstance(x, str) or isinstance(y, str):
        return None
    elif issequence(x) and issequence(y) and len(x) == len(y):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None


def is_variable(x):
    """A variable is an Expr with no args and a lowercase symbol as the op."""
    return isinstance(x, Expr) and not x.args and x.op[0].islower()


def unify_var(var, x, s):
    if var in s:
        return unify(s[var], x, s)
    elif x in s:
        return unify(var, s[x], s)
    elif occur_check(var, x, s):
        return None
    else:
        new_s = extend(s, var, x)
        cascade_substitution(new_s)
        return new_s


def occur_check(var, x, s):
    """Return true if variable var occurs anywhere in x
    (or in subst(s, x), if s has a binding for x)."""
    if var == x:
        return True
    elif is_variable(x) and x in s:
        return occur_check(var, s[x], s)
    elif isinstance(x, Expr):
        return (occur_check(var, x.op, s) or
                occur_check(var, x.args, s))
    elif isinstance(x, (list, tuple)):
        return first(e for e in x if occur_check(var, e, s))
    else:
        return False


def subst(s, x):
    """Substitute the substitution s into the expression x.
    >>> subst({x: 42, y:0}, F(x) + y)
    (F(42) + 0)
    """
    if isinstance(x, list):
        return [subst(s, xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([subst(s, xi) for xi in x])
    elif not isinstance(x, Expr):
        return x
    elif is_var_symbol(x.op):
        return s.get(x, x)
    else:
        return Expr(x.op, *[subst(s, arg) for arg in x.args])


def cascade_substitution(s):
    """This method allows to return a correct unifier in normal form
    and perform a cascade substitution to s.
    For every mapping in s perform a cascade substitution on s.get(x)
    and if it is replaced with a function ensure that all the function 
    terms are correct updates by passing over them again.
    >>> s = {x: y, y: G(z)}
    >>> cascade_substitution(s)
    >>> s == {x: G(z), y: G(z)}
    True
    """

    for x in s:
        s[x] = subst(s, s.get(x))
        if isinstance(s.get(x), Expr) and not is_variable(s.get(x)):
            # Ensure Function Terms are correct updates by passing over them again
            s[x] = subst(s, s.get(x))


def unify_mm(x, y, s={}):
    """Unify expressions x,y with substitution s using an efficient rule-based
    unification algorithm by Martelli & Montanari; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. Expr('x')), constants, lists, or Exprs.
    >>> unify_mm(x, 3, {})
    {x: 3}
    """

    set_eq = extend(s, x, y)
    s = set_eq.copy()
    while True:
        trans = 0
        for x, y in set_eq.items():
            if x == y:
                # if x = y this mapping is deleted (rule b)
                del s[x]
            elif not is_variable(x) and is_variable(y):
                # if x is not a variable and y is a variable, rewrite it as y = x in s (rule a)
                if s.get(y, None) is None:
                    s[y] = x
                    del s[x]
                else:
                    # if a mapping already exist for variable y then apply
                    # variable elimination (there is a chance to apply rule d)
                    s[x] = vars_elimination(y, s)
            elif not is_variable(x) and not is_variable(y):
                # in which case x and y are not variables, if the two root function symbols
                # are different, stop with failure, else apply term reduction (rule c)
                if x.op is y.op and len(x.args) == len(y.args):
                    term_reduction(x, y, s)
                    del s[x]
                else:
                    return None
            elif isinstance(y, Expr):
                # in which case x is a variable and y is a function or a variable (e.g. F(z) or y),
                # if y is a function, we must check if x occurs in y, then stop with failure, else
                # try to apply variable elimination to y (rule d)
                if occur_check(x, y, s):
                    return None
                s[x] = vars_elimination(y, s)
                if y == s.get(x):
                    trans += 1
            else:
                trans += 1
        if trans == len(set_eq):
            # if no transformation has been applied, stop with success
            return s
        set_eq = s.copy()


def term_reduction(x, y, s):
    """Apply term reduction to x and y if both are functions and the two root function
    symbols are equals (e.g. F(x1, x2, ..., xn) and F(x1', x2', ..., xn')) by returning
    a new mapping obtained by replacing x: y with {x1: x1', x2: x2', ..., xn: xn'}
    """
    for i in range(len(x.args)):
        if x.args[i] in s:
            s[s.get(x.args[i])] = y.args[i]
        else:
            s[x.args[i]] = y.args[i]


def vars_elimination(x, s):
    """Apply variable elimination to x: if x is a variable and occurs in s, return
    the term mapped by x, else if x is a function recursively applies variable
    elimination to each term of the function."""
    if not isinstance(x, Expr):
        return x
    if is_variable(x):
        return s.get(x, x)
    return Expr(x.op, *[vars_elimination(arg, s) for arg in x.args])


def standardize_variables(sentence, dic=None):
    """Replace all the variables in sentence with new variables."""
    if dic is None:
        dic = {}
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op):
        if sentence in dic:
            return dic[sentence]
        else:
            v = Expr('v_{}'.format(next(standardize_variables.counter)))
            dic[sentence] = v
            return v
    else:
        return Expr(sentence.op, *[standardize_variables(a, dic) for a in sentence.args])


standardize_variables.counter = itertools.count()


# ______________________________________________________________________________


def parse_clauses_from_dimacs(dimacs_cnf):
    """Converts a string into CNF clauses according to the DIMACS format used in SAT competitions"""
    return map(lambda c: associate('|', c),
               map(lambda c: [expr('~X' + str(abs(l))) if l < 0 else expr('X' + str(l)) for l in c],
                   map(lambda line: map(int, line.split()),
                       filter(None, ' '.join(
                           filter(lambda line: line[0] not in ('c', 'p'),
                                  filter(None, dimacs_cnf.strip().replace('\t', ' ').split('\n')))).split(' 0')))))


# ______________________________________________________________________________


class FolKB(KB):
    """A knowledge base consisting of first-order definite clauses.
    >>> kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'),
    ...              expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    >>> kb0.tell(expr('Rabbit(Flopsie)'))
    >>> kb0.retract(expr('Rabbit(Pete)'))
    >>> kb0.ask(expr('Hates(Mac, x)'))[x]
    Flopsie
    >>> kb0.ask(expr('Wife(Pete, x)'))
    False
    """

    def __init__(self, clauses=None):
        super().__init__()
        self.clauses = []  # inefficient: no indexing
        if clauses:
            for clause in clauses:
                self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception('Not a definite clause: {}'.format(sentence))

    def ask_generator(self, query):
        return fol_bc_ask(self, query)

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal):
        return self.clauses


def fol_fc_ask(kb, alpha):
    """
    [Figure 9.3]
    A simple forward-chaining algorithm.
    """
    # TODO: improve efficiency
    kb_consts = list({c for clause in kb.clauses for c in constant_symbols(clause)})

    def enum_subst(p):
        query_vars = list({v for clause in p for v in variables(clause)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    # check if we can answer without new inferences
    for q in kb.clauses:
        phi = unify_mm(q, alpha)
        if phi is not None:
            yield phi

    while True:
        new = []
        for rule in kb.clauses:
            p, q = parse_definite_clause(rule)
            for theta in enum_subst(p):
                if set(subst(theta, p)).issubset(set(kb.clauses)):
                    q_ = subst(theta, q)
                    if all([unify_mm(x, q_) is None for x in kb.clauses + new]):
                        new.append(q_)
                        phi = unify_mm(q_, alpha)
                        if phi is not None:
                            yield phi
        if not new:
            break
        for clause in new:
            kb.tell(clause)
    return None


def fol_bc_ask(kb, query):
    """
    [Figure 9.6]
    A simple backward-chaining algorithm for first-order logic.
    KB should be an instance of FolKB, and query an atomic sentence.
    """
    return fol_bc_or(kb, query, {})


def fol_bc_or(kb, goal, theta):
    for rule in kb.fetch_rules_for_goal(goal):
        lhs, rhs = parse_definite_clause(standardize_variables(rule))
        for theta1 in fol_bc_and(kb, lhs, unify_mm(rhs, goal, theta)):
            yield theta1


def fol_bc_and(kb, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(kb, subst(theta, first), theta):
            for theta2 in fol_bc_and(kb, rest, theta1):
                yield theta2


# A simple KB that defines the relevant conditions of the Wumpus World as in Figure 7.4.
# See Sec. 7.4.3
wumpus_kb = PropKB()

P11, P12, P21, P22, P31, B11, B21 = expr('P11, P12, P21, P22, P31, B11, B21')
wumpus_kb.tell(~P11)
wumpus_kb.tell(B11 | '<=>' | (P12 | P21))
wumpus_kb.tell(B21 | '<=>' | (P11 | P22 | P31))
wumpus_kb.tell(~B11)
wumpus_kb.tell(B21)

test_kb = FolKB(map(expr, ['Farmer(Mac)',
                           'Rabbit(Pete)',
                           'Mother(MrsMac, Mac)',
                           'Mother(MrsRabbit, Pete)',
                           '(Rabbit(r) & Farmer(f)) ==> Hates(f, r)',
                           '(Mother(m, c)) ==> Loves(m, c)',
                           '(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)',
                           '(Farmer(f)) ==> Human(f)',
                           # Note that this order of conjuncts
                           # would result in infinite recursion:
                           # '(Human(h) & Mother(m, h)) ==> Human(m)'
                           '(Mother(m, h) & Human(h)) ==> Human(m)']))

crime_kb = FolKB(map(expr, ['(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)',
                            'Owns(Nono, M1)',
                            'Missile(M1)',
                            '(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)',
                            'Missile(x) ==> Weapon(x)',
                            'Enemy(x, America) ==> Hostile(x)',
                            'American(West)',
                            'Enemy(Nono, America)']))


# ______________________________________________________________________________

# Example application (not in the book).
# You can use the Expr class to do symbolic differentiation. This used to be
# a part of AI; now it is considered a separate field, Symbolic Algebra.


def diff(y, x):
    """Return the symbolic derivative, dy/dx, as an Expr.
    However, you probably want to simplify the results with simp.
    >>> diff(x * x, x)
    ((x * 1) + (x * 1))
    """
    if y == x:
        return 1
    elif not y.args:
        return 0
    else:
        u, op, v = y.args[0], y.op, y.args[-1]
        if op == '+':
            return diff(u, x) + diff(v, x)
        elif op == '-' and len(y.args) == 1:
            return -diff(u, x)
        elif op == '-':
            return diff(u, x) - diff(v, x)
        elif op == '*':
            return u * diff(v, x) + v * diff(u, x)
        elif op == '/':
            return (v * diff(u, x) - u * diff(v, x)) / (v * v)
        elif op == '**' and isnumber(x.op):
            return v * u ** (v - 1) * diff(u, x)
        elif op == '**':
            return (v * u ** (v - 1) * diff(u, x) +
                    u ** v * Expr('log')(u) * diff(v, x))
        elif op == 'log':
            return diff(u, x) / u
        else:
            raise ValueError('Unknown op: {} in diff({}, {})'.format(op, y, x))


def simp(x):
    """Simplify the expression x."""
    if isnumber(x) or not x.args:
        return x
    args = list(map(simp, x.args))
    u, op, v = args[0], x.op, args[-1]
    if op == '+':
        if v == 0:
            return u
        if u == 0:
            return v
        if u == v:
            return 2 * u
        if u == -v or v == -u:
            return 0
    elif op == '-' and len(args) == 1:
        if u.op == '-' and len(u.args) == 1:
            return u.args[0]  # --y ==> y
    elif op == '-':
        if v == 0:
            return u
        if u == 0:
            return -v
        if u == v:
            return 0
        if u == -v or v == -u:
            return 0
    elif op == '*':
        if u == 0 or v == 0:
            return 0
        if u == 1:
            return v
        if v == 1:
            return u
        if u == v:
            return u ** 2
    elif op == '/':
        if u == 0:
            return 0
        if v == 0:
            return Expr('Undefined')
        if u == v:
            return 1
        if u == -v or v == -u:
            return 0
    elif op == '**':
        if u == 0:
            return 0
        if v == 0:
            return 1
        if u == 1:
            return 1
        if v == 1:
            return u
    elif op == 'log':
        if u == 1:
            return 0
    else:
        raise ValueError('Unknown op: ' + op)
    # If we fall through to here, we can not simplify further
    return Expr(op, *args)


def d(y, x):
    """Differentiate and then simplify.
    >>> d(x * x - x, x)
    ((2 * x) - 1)
    """
    return simp(diff(y, x))
