"""Representations and Inference for Logic (Chapters 7-10)

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
operator precedence of commas; you may need to add parents to make precedence work.
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
import itertools
import random
from collections import defaultdict

from agents import Agent, Glitter, Bump, Stench, Breeze, Scream
from search import astar_search, PlanRoute
from utils4e import remove_all, unique, first, probability, isnumber, issequence, Expr, expr, subexpressions


# ______________________________________________________________________________
# Chapter 7 Logical Agents
# 7.1 Knowledge Based Agents


class KB:
    """
    A knowledge base to which you can tell and ask sentences.
    To create a KB, subclass this class and implement tell, ask_generator, and retract.
    Ask_generator：
      For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
      FOL KB, something like ask(Brother(x, y)) might return many substitutions
      such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
      So ask_generator generates these one at a time, and ask either returns the
      first one or returns False.
    """

    def __init__(self, sentence=None):
        raise NotImplementedError

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
        self.clauses = []
        if sentence:
            self.tell(sentence)

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


def KB_AgentProgram(KB):
    """A generic logical knowledge-based agent program. [Figure 7.1]"""
    steps = itertools.count()

    def program(percept):
        t = next(steps)
        KB.tell(make_percept_sentence(percept, t))
        action = KB.ask(make_action_query(t))
        KB.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(percept, t):
        return Expr("Percept")(percept, t)

    def make_action_query(t):
        return expr("ShouldDo(action, {})".format(t))

    def make_action_sentence(action, t):
        return Expr("Did")(action[expr('action')], t)

    return program


# _____________________________________________________________________________
# 7.2 The Wumpus World


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
# 7.4 Propositional Logic


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
    """
    Returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive.  In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op) and
                all(is_symbol(arg.op) for arg in conjuncts(antecedent)))
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
A, B, C, D, E, F, G, P, Q, x, y, z = map(Expr, 'ABCDEFGPQxyz')


# ______________________________________________________________________________
# 7.4.4 A simple inference procedure


def tt_entails(kb, alpha):
    """
    Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. [Figure 7.10]. Note that the 'kb' should be an
    Expr which is a conjunction of clauses.
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
    """
    Return a set of (symbol_name, arity) in x.
    All symbols (even functional) with arity > 0 are considered.
    """
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
    """
    Return True if the propositional logic expression is true in the model,
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
        raise ValueError("illegal operator in logic expression" + str(exp))


# ______________________________________________________________________________
# 7.5 Propositional Theorem Proving


def to_cnf(s):
    """Convert a propositional logical sentence to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...) [p. 253]
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


def pl_resolution(KB, alpha):
    """
    Propositional-logic resolution: say if alpha follows from KB. [Figure 7.12]
    >>> pl_resolution(horn_clauses_KB, A)
    True
    """
    clauses = KB.clauses + conjuncts(to_cnf(~alpha))
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
                dnew = unique(remove_all(di, disjuncts(ci)) +
                              remove_all(dj, disjuncts(cj)))
                clauses.append(associate('|', dnew))
    return clauses


# ______________________________________________________________________________
# 7.5.4 Forward and backward chaining


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
        return [c for c in self.clauses
                if c.op == '==>' and p in conjuncts(c.args[0])]


def pl_fc_entails(KB, q):
    """Use forward chaining to see if a PropDefiniteKB entails symbol q.
    [Figure 7.15]
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """
    count = {c: len(conjuncts(c.args[0]))
             for c in KB.clauses
             if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in KB.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in KB.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False


""" [Figure 7.13]
Simple inference in a wumpus world example
"""
wumpus_world_inference = expr("(B11 <=> (P12 | P21))  &  ~B11")

""" [Figure 7.16]
Propositional Logic Forward Chaining example
"""
horn_clauses_KB = PropDefiniteKB()
for s in "P==>Q; (L&M)==>P; (B&L)==>M; (A&P)==>L; (A&B)==>L; A;B".split(';'):
    horn_clauses_KB.tell(expr(s))

"""
Definite clauses KB example
"""
definite_clauses_KB = PropDefiniteKB()
for clause in ['(B & F)==>E', '(A & E & F)==>G', '(B & C)==>F', '(A & B)==>D', '(E & F)==>H', '(H & I)==>J', 'A', 'B',
               'C']:
    definite_clauses_KB.tell(expr(clause))


# ______________________________________________________________________________
# 7.6 Effective Propositional Model Checking
# DPLL-Satisfiable [Figure 7.17]


def dpll_satisfiable(s):
    """Check satisfiability of a propositional sentence.
    This differs from the book code in two ways: (1) it returns a model
    rather than True when it succeeds; this is more useful. (2) The
    function find_pure_symbol is passed a list of unknown clauses, rather
    than a list of all clauses and the model; this is more efficient.
    >>> dpll_satisfiable(A |'<=>'| B) == {A: True, B: True}
    True
    """
    clauses = conjuncts(to_cnf(s))
    symbols = list(prop_symbols(s))
    return dpll(clauses, symbols, {})


def dpll(clauses, symbols, model):
    """See if the clauses are true in a partial model."""
    unknown_clauses = []  # clauses with an unknown truth value
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value))
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value))
    if not symbols:
        raise TypeError("Argument should be of the type Expr.")
    P, symbols = symbols[0], symbols[1:]
    return (dpll(clauses, symbols, extend(model, P, True)) or
            dpll(clauses, symbols, extend(model, P, False)))


def find_pure_symbol(symbols, clauses):
    """
    Find a symbol and its value if it appears only as a positive literal
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
    """
    Find a forced assignment if possible from a clause with only 1
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
# 7.6.2 Local search algorithms
# Walk-SAT [Figure 7.18]


def WalkSAT(clauses, p=0.5, max_flips=10000):
    """
    Checks for satisfiability of all clauses by randomly flipping values of variables
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
# 7.7 Agents Based on Propositional Logic
# 7.7.1 The current state of the world


class WumpusKB(PropKB):
    """
    Create a Knowledge Base that contains the atemporal "Wumpus physics" and temporal rules with time zero.
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

                s.append(
                    equiv(
                        location(i, j, time), location(i, j, time) & ~move_forward(time) | percept_bump(time)))

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
                self.tell(
                    equiv(ok_to_move(i, j, time), ~pit(i, j) & ~wumpus(i, j) & wumpus_alive(time))
                )

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
        if (other.get_location() == self.get_location() and
                other.get_orientation() == self.get_orientation()):
            return True
        else:
            return False


# ______________________________________________________________________________
# 7.7.2 A hybrid agent


class HybridWumpusAgent(Agent):
    """An agent for the wumpus world that does logical inference. [Figure 7.20]"""

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
# 7.7.4 Making plans by propositional inference


def SAT_plan(init, transition, goal, t_max, SAT_solver=dpll_satisfiable):
    """Converts a planning problem to Satisfaction problem by translating it to a cnf sentence.
    [Figure 7.22]
    >>> transition = {'A': {'Left': 'A', 'Right': 'B'}, 'B': {'Left': 'A', 'Right': 'C'}, 'C': {'Left': 'B', 'Right': 'C'}}
    >>> SAT_plan('A', transition, 'C', 2) is None
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
                state_sym[s, t] = Expr("State_{}".format(next(state_counter)))

        # Add initial state axiom
        clauses.append(state_sym[init, 0])

        # Add goal state axiom
        clauses.append(state_sym[goal, time])

        # All possible transitions
        transition_counter = itertools.count()
        for s in states:
            for action in transition[s]:
                s_ = transition[s][action]
                for t in range(time):
                    # Action 'action' taken from state 's' at time 't' to reach 's_'
                    action_sym[s, action, t] = Expr(
                        "Transition_{}".format(next(transition_counter)))

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
    for t in range(t_max):
        # dictionaries to help extract the solution from model
        state_sym = {}
        action_sym = {}

        cnf = translate_to_SAT(init, transition, goal, t)
        model = SAT_solver(cnf)
        if model is not False:
            return extract_solution(model)
    return None


# ______________________________________________________________________________
# Chapter 9 Inference in First Order Logic
# 9.2 Unification and First Order Inference
# 9.2.1 Unification


def unify(x, y, s={}):
    """Unify expressions x,y with substitution s; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. Expr('x')), constants, lists, or Exprs. [Figure 9.1]
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
        return extend(s, var, x)


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


def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val; return copy.
    >>> extend({x: 1}, y, 2) == {x: 1, y: 2}
    True
    """
    s2 = s.copy()
    s2[var] = val
    return s2


# 9.2.2 Storage and retrieval


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

    def __init__(self, initial_clauses=None):
        self.clauses = []  # inefficient: no indexing
        if initial_clauses:
            for clause in initial_clauses:
                self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception("Not a definite clause: {}".format(sentence))

    def ask_generator(self, query):
        return fol_bc_ask(self, query)

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal):
        return self.clauses


# ______________________________________________________________________________
# 9.3 Forward Chaining
# 9.3.2 A simple forward-chaining algorithm


def fol_fc_ask(KB, alpha):
    """A simple forward-chaining algorithm. [Figure 9.3]"""
    kb_consts = list({c for clause in KB.clauses for c in constant_symbols(clause)})

    def enum_subst(p):
        query_vars = list({v for clause in p for v in variables(clause)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    # check if we can answer without new inferences
    for q in KB.clauses:
        phi = unify(q, alpha, {})
        if phi is not None:
            yield phi

    while True:
        new = []
        for rule in KB.clauses:
            p, q = parse_definite_clause(rule)
            for theta in enum_subst(p):
                if set(subst(theta, p)).issubset(set(KB.clauses)):
                    q_ = subst(theta, q)
                    if all([unify(x, q_, {}) is None for x in KB.clauses + new]):
                        new.append(q_)
                        phi = unify(q_, alpha, {})
                        if phi is not None:
                            yield phi
        if not new:
            break
        for clause in new:
            KB.tell(clause)
    return None


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
        return Expr(sentence.op,
                    *[standardize_variables(a, dic) for a in sentence.args])


standardize_variables.counter = itertools.count()


# __________________________________________________________________
# 9.4 Backward Chaining


def fol_bc_ask(KB, query):
    """A simple backward-chaining algorithm for first-order logic. [Figure 9.6]
    KB should be an instance of FolKB, and query an atomic sentence."""
    return fol_bc_or(KB, query, {})


def fol_bc_or(KB, goal, theta):
    for rule in KB.fetch_rules_for_goal(goal):
        lhs, rhs = parse_definite_clause(standardize_variables(rule))
        for theta1 in fol_bc_and(KB, lhs, unify(rhs, goal, theta)):
            yield theta1


def fol_bc_and(KB, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(KB, subst(theta, first), theta):
            for theta2 in fol_bc_and(KB, rest, theta1):
                yield theta2


# ______________________________________________________________________________
# A simple KB that defines the relevant conditions of the Wumpus World as in Fig 7.4.
# See Sec. 7.4.3
wumpus_kb = PropKB()

P11, P12, P21, P22, P31, B11, B21 = expr('P11, P12, P21, P22, P31, B11, B21')
wumpus_kb.tell(~P11)
wumpus_kb.tell(B11 | '<=>' | (P12 | P21))
wumpus_kb.tell(B21 | '<=>' | (P11 | P22 | P31))
wumpus_kb.tell(~B11)
wumpus_kb.tell(B21)

test_kb = FolKB(
    map(expr, ['Farmer(Mac)',
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

crime_kb = FolKB(
    map(expr, ['(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)',
               'Owns(Nono, M1)',
               'Missile(M1)',
               '(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)',
               'Missile(x) ==> Weapon(x)',
               'Enemy(x, America) ==> Hostile(x)',
               'American(West)',
               'Enemy(Nono, America)']))


# ______________________________________________________________________________

# Example application (not in the book).
# You can use the Expr class to do symbolic differentiation.  This used to be
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
            return (v * u ** (v - 1) * diff(u, x))
        elif op == '**':
            return (v * u ** (v - 1) * diff(u, x) +
                    u ** v * Expr('log')(u) * diff(v, x))
        elif op == 'log':
            return diff(u, x) / u
        else:
            raise ValueError("Unknown op: {} in diff({}, {})".format(op, y, x))


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
        raise ValueError("Unknown op: " + op)
    # If we fall through to here, we can not simplify further
    return Expr(op, *args)


def d(y, x):
    """Differentiate and then simplify.
    >>> d(x * x - x, x)
    ((2 * x) - 1)
    """
    return simp(diff(y, x))
