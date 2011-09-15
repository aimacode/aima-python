"""Representations and Inference for Logic (Chapters 7-10)

Covers both Propositional and First-Order Logic. First we have four
important data types:

    KB            Abstract class holds a knowledge base of logical expressions
    KB_Agent      Abstract class subclasses agents.Agent
    Expr          A logical expression
    substitution  Implemented as a dictionary of var:value pairs, {x:1, y:x}

Be careful: some functions take an Expr as argument, and some take a KB.
Then we implement various functions for doing logical inference:

    pl_true          Evaluate a propositional logical sentence in a model
    tt_entails       Say if a statement is entailed by a KB
    pl_resolution    Do resolution on propositional sentences
    dpll_satisfiable See if a propositional sentence is satisfiable
    WalkSAT          (not yet implemented)

And a few other functions:

    to_cnf           Convert to conjunctive normal form
    unify            Do unification of two FOL sentences
    diff, simp       Symbolic differentiation and simplification             
"""

from __future__ import generators
import re
import agents
from utils import *

#______________________________________________________________________________

class KB:
    """A Knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract.  Why ask_generator instead of ask?  
    The book is a bit vague on what ask means --
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
    FOL KB, something like ask(Brother(x, y)) might return many substitutions
    such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.  
    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        abstract

    def tell(self, sentence): 
        "Add the sentence to the KB"
        abstract

    def ask(self, query):
        """Ask returns a substitution that makes the query true, or
        it returns False. It is implemented in terms of ask_generator."""
        try: 
            return self.ask_generator(query).next()
        except StopIteration:
            return False

    def ask_generator(self, query): 
        "Yield all the substitutions that make query true."
        abstract

    def retract(self, sentence):
        "Remove the sentence from the KB"
        abstract


class PropKB(KB):
    "A KB for Propositional Logic.  Inefficient, with no indexing."

    def __init__(self, sentence=None):
        self.clauses = []
        if sentence:
            self.tell(sentence)

    def tell(self, sentence): 
        "Add the sentence's clauses to the KB"
        self.clauses.extend(conjuncts(to_cnf(sentence)))        

    def ask_generator(self, query): 
        "Yield the empty substitution if KB implies query; else False"
        if not tt_entails(Expr('&', *self.clauses), query):
            return
        yield {}

    def retract(self, sentence):
        "Remove the sentence's clauses from the KB"
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)

#______________________________________________________________________________
    
class KB_Agent(agents.Agent):
    """A generic logical knowledge-based agent. [Fig. 7.1]"""
    def __init__(self, KB):
        t = 0
        def program(percept):
            KB.tell(self.make_percept_sentence(percept, t))
            action = KB.ask(self.make_action_query(t))
            KB.tell(self.make_action_sentence(action, t))
            t = t + 1
            return action
        self.program = program

    def make_percept_sentence(self, percept, t): 
        return(Expr("Percept")(percept, t))

    def make_action_query(self, t): 
        return(expr("ShouldDo(action, %d)" % t))

    def make_action_sentence(self, action, t):
        return(Expr("Did")(action, t))

#______________________________________________________________________________

class Expr:
    """A symbolic mathematical expression.  We use this class for logical
    expressions, and for terms within logical expressions. In general, an
    Expr has an op (operator) and a list of args.  The op can be:
      Null-ary (no args) op:
        A number, representing the number itself.  (e.g. Expr(42) => 42)
        A symbol, representing a variable or constant (e.g. Expr('F') => F)
      Unary (1 arg) op:
        '~', '-', representing NOT, negation (e.g. Expr('~', Expr('P')) => ~P)
      Binary (2 arg) op:
        '>>', '<<', representing forward and backward implication
        '+', '-', '*', '/', '**', representing arithmetic operators
        '<', '>', '>=', '<=', representing comparison operators
        '<=>', '^', representing logical equality and XOR
      N-ary (0 or more args) op:
        '&', '|', representing conjunction and disjunction
        A symbol, representing a function term or FOL proposition

    Exprs can be constructed with operator overloading: if x and y are Exprs,
    then so are x + y and x & y, etc.  Also, if F and x are Exprs, then so is 
    F(x); it works by overloading the __call__ method of the Expr F.  Note 
    that in the Expr that is created by F(x), the op is the str 'F', not the 
    Expr F.   See http://www.python.org/doc/current/ref/specialnames.html 
    to learn more about operator overloading in Python.

    WARNING: x == y and x != y are NOT Exprs.  The reason is that we want
    to write code that tests 'if x == y:' and if x == y were the same
    as Expr('==', x, y), then the result would always be true; not what a
    programmer would expect.  But we still need to form Exprs representing
    equalities and disequalities.  We concentrate on logical equality (or
    equivalence) and logical disequality (or XOR).  You have 3 choices:
        (1) Expr('<=>', x, y) and Expr('^', x, y)
            Note that ^ is bitwose XOR in Python (and Java and C++)
        (2) expr('x <=> y') and expr('x =/= y').  
            See the doc string for the function expr.
        (3) (x % y) and (x ^ y).
            It is very ugly to have (x % y) mean (x <=> y), but we need
            SOME operator to make (2) work, and this seems the best choice.

    WARNING: if x is an Expr, then so is x + 1, because the int 1 gets
    coerced to an Expr by the constructor.  But 1 + x is an error, because
    1 doesn't know how to add an Expr.  (Adding an __radd__ method to Expr
    wouldn't help, because int.__add__ is still called first.) Therefore,
    you should use Expr(1) + x instead, or ONE + x, or expr('1 + x').
    """

    def __init__(self, op, *args):
        "Op is a string or number; args are Exprs (or are coerced to Exprs)."
        assert isinstance(op, str) or (isnumber(op) and not args)
        self.op = num_or_str(op)
        self.args = map(expr, args) ## Coerce args to Exprs

    def __call__(self, *args):
        """Self must be a symbol with no args, such as Expr('F').  Create a new
        Expr with 'F' as op and the args as arguments."""
        assert is_symbol(self.op) and not self.args
        return Expr(self.op, *args)

    def __repr__(self):
        "Show something like 'P' or 'P(x, y)', or '~P' or '(P | Q | R)'"
        if len(self.args) == 0: # Constant or proposition with arity 0
            return str(self.op)
        elif is_symbol(self.op): # Functional or Propositional operator
            return '%s(%s)' % (self.op, ', '.join(map(repr, self.args)))
        elif len(self.args) == 1: # Prefix operator
            return self.op + repr(self.args[0])
        else: # Infix operator
            return '(%s)' % (' '+self.op+' ').join(map(repr, self.args))

    def __eq__(self, other):
        """x and y are equal iff their ops and args are equal."""
        return (other is self) or (isinstance(other, Expr) 
            and self.op == other.op and self.args == other.args)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        "Need a hash method so Exprs can live in dicts."
        return hash(self.op) ^ hash(tuple(self.args))

    # See http://www.python.org/doc/current/lib/module-operator.html
    # Not implemented: not, abs, pos, concat, contains, *item, *slice
    def __lt__(self, other):     return Expr('<',  self, other)
    def __le__(self, other):     return Expr('<=', self, other)
    def __ge__(self, other):     return Expr('>=', self, other)
    def __gt__(self, other):     return Expr('>',  self, other)
    def __add__(self, other):    return Expr('+',  self, other)
    def __sub__(self, other):    return Expr('-',  self, other)
    def __and__(self, other):    return Expr('&',  self, other)
    def __div__(self, other):    return Expr('/',  self, other)
    def __truediv__(self, other):return Expr('/',  self, other)
    def __invert__(self):        return Expr('~',  self)
    def __lshift__(self, other): return Expr('<<', self, other)
    def __rshift__(self, other): return Expr('>>', self, other)
    def __mul__(self, other):    return Expr('*',  self, other)
    def __neg__(self):           return Expr('-',  self)
    def __or__(self, other):     return Expr('|',  self, other)
    def __pow__(self, other):    return Expr('**', self, other)
    def __xor__(self, other):    return Expr('^',  self, other)
    def __mod__(self, other):    return Expr('<=>',  self, other) ## (x % y)
    


def expr(s):
    """Create an Expr representing a logic expression by parsing the input
    string. Symbols and numbers are automatically converted to Exprs.
    In addition you can use alternative spellings of these operators:
      'x ==> y'   parses as   (x >> y)    # Implication
      'x <== y'   parses as   (x << y)    # Reverse implication
      'x <=> y'   parses as   (x % y)     # Logical equivalence
      'x =/= y'   parses as   (x ^ y)     # Logical disequality (xor)
    But BE CAREFUL; precedence of implication is wrong. expr('P & Q ==> R & S')
    is ((P & (Q >> R)) & S); so you must use expr('(P & Q) ==> (R & S)').
    >>> expr('P <=> Q(1)')
    (P <=> Q(1))
    >>> expr('P & Q | ~R(x, F(x))')
    ((P & Q) | ~R(x, F(x)))
    """
    if isinstance(s, Expr): return s
    if isnumber(s): return Expr(s)
    ## Replace the alternative spellings of operators with canonical spellings
    s = s.replace('==>', '>>').replace('<==', '<<')
    s = s.replace('<=>', '%').replace('=/=', '^')
    ## Replace a symbol or number, such as 'P' with 'Expr("P")'
    s = re.sub(r'([a-zA-Z0-9_.]+)', r'Expr("\1")', s)
    ## Now eval the string.  (A security hole; do not use with an adversary.)
    return eval(s, {'Expr':Expr})

def is_symbol(s):
    "A string s is a symbol if it starts with an alphabetic char."
    return isinstance(s, str) and s[0].isalpha()

def is_var_symbol(s):
    "A logic variable symbol is an initial-lowercase string."
    return is_symbol(s) and s[0].islower()

def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string other than
    TRUE or FALSE."""
    return is_symbol(s) and s[0].isupper() and s != 'TRUE' and s != 'FALSE'

def is_positive(s):
    """s is an unnegated logical expression
    >>> is_positive(expr('F(A, B)'))
    True
    >>> is_positive(expr('~F(A, B)'))
    False
    """
    return s.op != '~'

def is_negative(s):
    """s is a negated logical expression
    >>> is_negative(expr('F(A, B)'))
    False
    >>> is_negative(expr('~F(A, B)'))
    True
    """
    return s.op == '~'

def is_literal(s):
    """s is a FOL literal
    >>> is_literal(expr('~F(A, B)'))
    True
    >>> is_literal(expr('F(A, B)'))
    True
    >>> is_literal(expr('F(A, B) & G(B, C)'))
    False
    """
    return is_symbol(s.op) or (s.op == '~' and is_literal(s.args[0]))

def literals(s):
    """returns the list of literals of logical expression s.
    >>> literals(expr('F(A, B)'))
    [F(A, B)]
    >>> literals(expr('~F(A, B)'))
    [~F(A, B)]
    >>> literals(expr('(F(A, B) & G(B, C)) ==> R(A, C)'))
    [F(A, B), G(B, C), R(A, C)]
    """
    op = s.op
    if op in set(['&', '|', '<<', '>>', '%', '^']):
        result = []
        for arg in s.args:
            result.extend(literals(arg))
        return result
    elif is_literal(s):
        return [s]
    else:
        return []

def variables(s):
    """returns the set of variables in logical expression s.
    >>> ppset(variables(F(x, A, y)))
    set([x, y])
    >>> ppset(variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, z)')))
    set([x, y, z])
    """
    if is_literal(s):
        return set([v for v in s.args if is_variable(v)])
    else:
        vars = set([])
        for lit in literals(s):
            vars = vars.union(variables(lit))
        return vars
    
def is_definite_clause(s):
    """returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive.  In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    >>> is_definite_clause(expr('~Farmer(Mac)'))
    False
    >>> is_definite_clause(expr('(Farmer(f) & Rabbit(r)) ==> Hates(f, r)'))
    True
    >>> is_definite_clause(expr('(Farmer(f) & ~Rabbit(r)) ==> Hates(f, r)'))
    False
    """
    op = s.op
    return (is_symbol(op) or
            (op == '>>' and every(is_positive, literals(s))))

## Useful constant Exprs used in examples and code:
TRUE, FALSE, ZERO, ONE, TWO = map(Expr, ['TRUE', 'FALSE', 0, 1, 2]) 
A, B, C, F, G, P, Q, x, y, z  = map(Expr, 'ABCFGPQxyz') 

#______________________________________________________________________________

def tt_entails(kb, alpha):
    """Use truth tables to determine if KB entails sentence alpha. [Fig. 7.10]
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    return tt_check_all(kb, alpha, prop_symbols(kb & alpha), {})

def tt_check_all(kb, alpha, symbols, model):
    "Auxiliary routine to implement tt_entails."
    if not symbols:
        if pl_true(kb, model): return pl_true(alpha, model)
        else: return True
        assert result != None
    else:
        P, rest = symbols[0], symbols[1:]
        return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
                tt_check_all(kb, alpha, rest, extend(model, P, False)))

def prop_symbols(x):
    "Return a list of all propositional symbols in x."
    if not isinstance(x, Expr):
        return []
    elif is_prop_symbol(x.op):
        return [x]
    else:
        s = set(())
        for arg in x.args:
            for symbol in prop_symbols(arg):
                s.add(symbol)
        return list(s)

def tt_true(alpha):
    """Is the sentence alpha a tautology? (alpha will be coerced to an expr.)
    >>> tt_true(expr("(P >> Q) <=> (~P | Q)"))
    True
    """
    return tt_entails(TRUE, expr(alpha))

def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological."""
    op, args = exp.op, exp.args
    if exp == TRUE:
        return True
    elif exp == FALSE:
        return False
    elif is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p == None: return None
        else: return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p == True: return True
            if p == None: result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p == False: return False
            if p == None: result = None
        return result
    p, q = args
    if op == '>>':
        return pl_true(~p | q, model)
    elif op == '<<':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt == None: return None
    qt = pl_true(q, model)
    if qt == None: return None
    if op == '<=>':
        return pt == qt
    elif op == '^':
        return pt != qt
    else:
        raise ValueError, "illegal operator in logic expression" + str(exp)

#______________________________________________________________________________

## Convert to Conjunctive Normal Form (CNF)
 
def to_cnf(s):
    """Convert a propositional logical sentence s to conjunctive normal form.
    That is, of the form ((A | ~B | ...) & (B | C | ...) & ...) [p. 215]
    >>> to_cnf("~(B|C)")
    (~B & ~C)
    >>> to_cnf("B <=> (P1|P2)")
    ((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))
    >>> to_cnf("a | (b & c) | d")
    ((b | a | d) & (c | a | d))
    >>> to_cnf("A & (B | (D & E))")
    (A & (D | B) & (E | B))
    """
    if isinstance(s, str): s = expr(s)
    s = eliminate_implications(s) # Steps 1, 2 from p. 215
    s = move_not_inwards(s) # Step 3
    return distribute_and_over_or(s) # Step 4
    
def eliminate_implications(s):
    """Change >>, <<, and <=> into &, |, and ~. That is, return an Expr
    that is equivalent to s, but has only &, |, and ~ as logical operators.
    >>> eliminate_implications(A >> (~B << C))
    ((~B | ~C) | ~A)
    """
    if not s.args or is_symbol(s.op): return s     ## (Atoms are unchanged.)
    args = map(eliminate_implications, s.args)
    a, b = args[0], args[-1]
    if s.op == '>>':
        return (b | ~a)
    elif s.op == '<<':
        return (a | ~b)
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    else:
        return Expr(s.op, *args)

def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    >>> move_not_inwards(~(A & B))
    (~A | ~B)
    >>> move_not_inwards(~(~(A | ~B) | ~~C))
    ((A | ~B) & ~C)
    """
    if s.op == '~':
        NOT = lambda b: move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~': return move_not_inwards(a.args[0]) # ~~A ==> A
        if a.op =='&': return NaryExpr('|', *map(NOT, a.args))
        if a.op =='|': return NaryExpr('&', *map(NOT, a.args))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *map(move_not_inwards, s.args))

def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    if s.op == '|':
        s = NaryExpr('|', *s.args)
        if len(s.args) == 0: 
            return FALSE
        if len(s.args) == 1: 
            return distribute_and_over_or(s.args[0])
        conj = find_if((lambda d: d.op == '&'), s.args)
        if not conj:
            return NaryExpr(s.op, *s.args)
        others = [a for a in s.args if a is not conj]
        if len(others) == 1:
            rest = others[0]
        else:
            rest = NaryExpr('|', *others)
        return NaryExpr('&', *map(distribute_and_over_or,
                                  [(c|rest) for c in conj.args]))
    elif s.op == '&':
        return NaryExpr('&', *map(distribute_and_over_or, s.args))
    else:
        return s

_NaryExprTable = {'&':TRUE, '|':FALSE, '+':ZERO, '*':ONE}

def NaryExpr(op, *args):
    """Create an Expr, but with an nary, associative op, so we can promote
    nested instances of the same op up to the top level.
    >>> NaryExpr('&', (A&B),(B|C),(B&C))
    (A & B & (B | C) & B & C)
    """
    arglist = []
    for arg in args:
        if arg.op == op: arglist.extend(arg.args)
        else: arglist.append(arg)
    if len(args) == 1:
        return args[0]
    elif len(args) == 0:
        return _NaryExprTable[op]
    else:
        return Expr(op, *arglist)

def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    if isinstance(s, Expr) and s.op == '&': 
        return s.args
    else:
        return [s]

def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    if isinstance(s, Expr) and s.op == '|': 
        return s.args
    else:
        return [s]

#______________________________________________________________________________

def pl_resolution(KB, alpha):
    "Propositional Logic Resolution: say if alpha follows from KB. [Fig. 7.12]"
    clauses = KB.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i+1, n)]
        for (ci, cj) in pairs:
            resolvents = pl_resolve(ci, cj)
            if FALSE in resolvents: return True
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)): return False
        for c in new:
            if c not in clauses: clauses.append(c)

def pl_resolve(ci, cj):
    """Return all clauses that can be obtained by resolving clauses ci and cj.
    >>> for res in pl_resolve(to_cnf(A|B|C), to_cnf(~B|~C|F)):
    ...    ppset(disjuncts(res))
    set([A, C, F, ~C])
    set([A, B, F, ~B])
    """
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                dnew = unique(removeall(di, disjuncts(ci)) + 
                              removeall(dj, disjuncts(cj)))
                clauses.append(NaryExpr('|', *dnew))
    return clauses

#______________________________________________________________________________

class PropHornKB(PropKB):
    "A KB of Propositional Horn clauses."

    def tell(self, sentence):
        "Add a Horn Clauses to this KB."
        op = sentence.op
        assert op == '>>' or is_prop_symbol(op), "Must be Horn Clause"
        self.clauses.append(sentence)

    def ask_generator(self, query): 
        "Yield the empty substitution if KB implies query; else False"
        if not pl_fc_entails(self.clauses, query):
            return
        yield {}

    def retract(self, sentence):
        "Remove the sentence's clauses from the KB"
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)

    def clauses_with_premise(self, p):
        """The list of clauses in KB that have p in the premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses 
                if c.op == '>>' and p in conjuncts(c.args[0])]

def pl_fc_entails(KB, q):
    """Use forward chaining to see if a HornKB entails symbol q. [Fig. 7.14]
    >>> pl_fc_entails(Fig[7,15], expr('Q'))
    True
    """
    count = dict([(c, len(conjuncts(c.args[0]))) for c in KB.clauses
                                                 if c.op == '>>'])
    inferred = DefaultDict(False)
    agenda = [s for s in KB.clauses if is_prop_symbol(s.op)]
    if q in agenda: return True
    while agenda:
        p = agenda.pop()
        if not inferred[p]:
            inferred[p] = True
            for c in KB.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    if c.args[1] == q: return True
                    agenda.append(c.args[1])
    return False

## Wumpus World example [Fig. 7.13]
Fig[7,13] = expr("(B11 <=> (P12 | P21))  &  ~B11")

## Propositional Logic Forward Chaining example [Fig. 7.15]
Fig[7,15] = PropHornKB()
for s in "P>>Q   (L&M)>>P   (B&L)>>M   (A&P)>>L   (A&B)>>L   A   B".split(): 
    Fig[7,15].tell(expr(s))

#______________________________________________________________________________

# DPLL-Satisfiable [Fig. 7.16]

def dpll_satisfiable(s):
    """Check satisfiability of a propositional sentence.
    This differs from the book code in two ways: (1) it returns a model
    rather than True when it succeeds; this is more useful. (2) The
    function find_pure_symbol is passed a list of unknown clauses, rather
    than a list of all clauses and the model; this is more efficient.
    >>> ppsubst(dpll_satisfiable(A&~B))
    {A: True, B: False}
    >>> dpll_satisfiable(P&~P)
    False
    """
    clauses = conjuncts(to_cnf(s))
    symbols = prop_symbols(s)
    return dpll(clauses, symbols, {})
 
def dpll(clauses, symbols, model):
    "See if the clauses are true in a partial model."
    unknown_clauses = [] ## clauses with an unknown truth value
    for c in clauses:
        val =  pl_true(c, model)
        if val == False:
            return False
        if val != True: 
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    P = symbols.pop()
    return (dpll(clauses, symbols, extend(model, P, True)) or
            dpll(clauses, symbols, extend(model, P, False)))
 
def find_pure_symbol(symbols, unknown_clauses):
    """Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.
    >>> find_pure_symbol([A, B, C], [A|~B,~B|~C,C|A])
    (A, True)
    """
    for s in symbols:
        found_pos, found_neg = False, False
        for c in unknown_clauses:
            if not found_pos and s in disjuncts(c): found_pos = True
            if not found_neg and ~s in disjuncts(c): found_neg = True
        if found_pos != found_neg: return s, found_pos
    return None, None

def find_unit_clause(clauses, model):
    """A unit clause has only 1 variable that is not bound in the model.
    >>> find_unit_clause([A|B|C, B|~C, A|~B], {A:True})
    (B, False)
    """
    for clause in clauses:
        num_not_in_model = 0
        for literal in disjuncts(clause):
            sym = literal_symbol(literal)
            if sym not in model:
                num_not_in_model += 1
                P, value = sym, (literal.op != '~')
        if num_not_in_model == 1:
            return P, value
    return None, None
                

def literal_symbol(literal):
    """The symbol in this literal (without the negation).
    >>> literal_symbol(P)
    P
    >>> literal_symbol(~P)
    P
    """
    if literal.op == '~':
        return literal.args[0]
    else:
        return literal
        

#______________________________________________________________________________
# Walk-SAT [Fig. 7.17]

def WalkSAT(clauses, p=0.5, max_flips=10000):
    ## model is a random assignment of true/false to the symbols in clauses
    ## See ~/aima1e/print1/manual/knowledge+logic-answers.tex ???
    model = dict([(s, random.choice([True, False])) 
                 for s in prop_symbols(clauses)])
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            if_(pl_true(clause, model), satisfied, unsatisfied).append(clause)
        if not unsatisfied: ## if model satisfies all the clauses
            return model
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(prop_symbols(clause))
        else:
            ## Flip the symbol in clause that maximizes number of sat. clauses
            raise NotImplementedError
        model[sym] = not model[sym]


# PL-Wumpus-Agent [Fig. 7.19]
class PLWumpusAgent(agents.Agent):
    "An agent for the wumpus world that does logical inference. [Fig. 7.19]"""
    def __init__(self):
        KB = FOLKB() ## shouldn't this be a propositional KB? ***
        x, y, orientation = 1, 1, (1, 0)
        visited = set() ## squares already visited
        action = None
        plan = []

        def program(percept):
            stench, breeze, glitter = percept
            x, y, orientation = update_position(x, y, orientation, action)
            KB.tell('%sS_%d,%d' % (if_(stench, '', '~'), x, y))
            KB.tell('%sB_%d,%d' % (if_(breeze, '', '~'), x, y))
            if glitter: action = 'Grab'
            elif plan: action = plan.pop()
            else:
                for [i, j] in fringe(visited):
                    if KB.ask('~P_%d,%d & ~W_%d,%d' % (i, j, i, j)) != False:
                        raise NotImplementedError
                    KB.ask('~P_%d,%d | ~W_%d,%d' % (i, j, i, j)) != False 
            if action == None: 
                action = random.choice(['Forward', 'Right', 'Left'])
            return action

        self.program = program

def update_position(x, y, orientation, action):
    if action == 'TurnRight':
        orientation = turn_right(orientation)
    elif action == 'TurnLeft':
        orientation = turn_left(orientation)
    elif action == 'Forward':
        x, y = x + vector_add((x, y), orientation)
    return x, y, orientation
    
#______________________________________________________________________________

def unify(x, y, s):
    """Unify expressions x,y with substitution s; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. Expr('x')), constants, lists, or Exprs. [Fig. 9.1]
    >>> ppsubst(unify(x + y, y + C, {}))
    {x: y, y: C}
    """
    if s == None:
        return None
    elif x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif isinstance(x, Expr) and isinstance(y, Expr):
        return unify(x.args, y.args, unify(x.op, y.op, s))
    elif isinstance(x, str) or isinstance(y, str) or not x or not y:
        # orig. return if_(x == y, s, None) but we already know x != y
        return None
    elif issequence(x) and issequence(y) and len(x) == len(y):
        # Assert neither x nor y is []
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None

def is_variable(x):
    "A variable is an Expr with no args and a lowercase symbol as the op."
    return isinstance(x, Expr) and not x.args and is_var_symbol(x.op)

def unify_var(var, x, s):
    if var in s:
        return unify(s[var], x, s)
    elif occur_check(var, x, s):
        return None
    else:
        return extend(s, var, x)

def occur_check(var, x, s):
    """Return true if variable var occurs anywhere in x
    (or in subst(s, x), if s has a binding for x)."""

    if var == x:
        return True
    elif is_variable(x) and s.has_key(x):
        return occur_check(var, s[x], s) # fixed
    # What else might x be?  an Expr, a list, a string?
    elif isinstance(x, Expr):
        # Compare operator and arguments
        return (occur_check(var, x.op, s) or
                occur_check(var, x.args, s))
    elif isinstance(x, list) and x != []:
        # Compare first and rest
        return (occur_check(var, x[0], s) or
                occur_check(var, x[1:], s))
    else:
        # A variable cannot occur in a string
        return False
    
    #elif isinstance(x, Expr):
    #    return var.op == x.op or occur_check(var, x.args)
    #elif not isinstance(x, str) and issequence(x):
    #    for xi in x:
    #        if occur_check(var, xi): return True
    #return False

def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val;
    return copy.
    
    >>> ppsubst(extend({x: 1}, y, 2))
    {x: 1, y: 2}
    """
    s2 = s.copy()
    s2[var] = val
    return s2
    
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
        
def fol_fc_ask(KB, alpha):
    """Inefficient forward chaining for first-order logic. [Fig. 9.3]
    KB is an FOLHornKB and alpha must be an atomic sentence."""
    while True:
        new = {}
        for r in KB.clauses:
            r1 = standardize_apart(r)
            ps, q = conjuncts(r.args[0]), r.args[1]
            raise NotImplementedError

def standardize_apart(sentence, dic={}):
    """Replace all the variables in sentence with new variables.
    >>> e = expr('F(a, b, c) & G(c, A, 23)')
    >>> len(variables(standardize_apart(e)))
    3
    >>> variables(e).intersection(variables(standardize_apart(e)))
    set([])
    >>> is_variable(standardize_apart(expr('x')))
    True
    """
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op): 
        if sentence in dic:
            return dic[sentence]
        else:
            standardize_apart.counter += 1
            v = Expr('v_%d' % standardize_apart.counter)
            dic[sentence] = v
            return v
    else: 
        return Expr(sentence.op,
                    *[standardize_apart(a, dic) for a in sentence.args])

standardize_apart.counter = 0

#______________________________________________________________________________


class FolKB (KB):
    """A knowledge base consisting of first-order definite clauses
    >>> kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'),
    ...              expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    >>> kb0.tell(expr('Rabbit(Flopsie)'))
    >>> kb0.retract(expr('Rabbit(Pete)'))
    >>> kb0.ask(expr('Hates(Mac, x)'))[x]
    Flopsie
    >>> kb0.ask(expr('Wife(Pete, x)'))
    False
    """

    def __init__(self, initial_clauses=[]):
        self.clauses = [] # inefficient: no indexing
        for clause in initial_clauses:
            self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception("Not a definite clause: %s" % sentence)

    def ask_generator(self, query):
        return fol_bc_ask(self, [query])

    def retract(self, sentence):
        self.clauses.remove(sentence)

def test_ask(q):
    e = expr(q)
    vars = variables(e)
    ans = fol_bc_ask(test_kb, [e])
    res = []
    for a in ans:
        res.append(pretty(dict([(x, v) for (x, v) in a.items() if x in vars])))
    res.sort(key=str)
    return res

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
               #'(Human(h) & Mother(m, h)) ==> Human(m)'
               '(Mother(m, h) & Human(h)) ==> Human(m)'
               ])
)
               
    
def fol_bc_ask(KB, goals, theta={}):
    """A simple backward-chaining algorithm for first-order logic. [Fig. 9.6]
    KB should be an instance of FolKB, and goals a list of literals.

    >>> test_ask('Farmer(x)')
    ['{x: Mac}']
    >>> test_ask('Human(x)')
    ['{x: Mac}', '{x: MrsMac}']
    >>> test_ask('Hates(x, y)')
    ['{x: Mac, y: Pete}']
    >>> test_ask('Loves(x, y)')
    ['{x: MrsMac, y: Mac}', '{x: MrsRabbit, y: Pete}']
    >>> test_ask('Rabbit(x)')
    ['{x: MrsRabbit}', '{x: Pete}']
    """

    if goals == []:
        yield theta
        raise StopIteration()
    
    q1 = subst(theta, goals[0])

    for r in KB.clauses:
        sar = standardize_apart(r)
        
        # Split into head and body
        if is_symbol(sar.op):
            head = sar
            body = []
        elif sar.op == '>>': # sar = (Body1 & Body2 & ...) >> Head
            head = sar.args[1]
            body = sar.args[0] # as conjunction
        else:
            raise Exception("Invalid clause in FolKB: %s" % r)

        theta1 = unify(head, q1, {})

        if theta1 is not None:
            if body == []:
                new_goals = goals[1:]
            else:
                new_goals = conjuncts(body) + goals[1:]
            
            for ans in fol_bc_ask(KB, new_goals, subst_compose(theta1, theta)):
                yield ans

    raise StopIteration()

def subst_compose (s1, s2):
    """Return the substitution which is equivalent to applying s2 to
    the result of applying s1 to an expression.

    >>> s1 = {x: A, y: B}
    >>> s2 = {z: x, x: C}
    >>> p = F(x) & G(y) & expr('H(z)')
    >>> subst(s1, p)
    ((F(A) & G(B)) & H(z))
    >>> subst(s2, p)
    ((F(C) & G(y)) & H(x))
    
    >>> subst(s2, subst(s1, p))
    ((F(A) & G(B)) & H(x))
    >>> subst(subst_compose(s1, s2), p)
    ((F(A) & G(B)) & H(x))

    >>> subst(s1, subst(s2, p))
    ((F(C) & G(B)) & H(A))
    >>> subst(subst_compose(s2, s1), p)
    ((F(C) & G(B)) & H(A))
    >>> ppsubst(subst_compose(s1, s2))
    {x: A, y: B, z: x}
    >>> ppsubst(subst_compose(s2, s1))
    {x: C, y: B, z: A}
    >>> subst(subst_compose(s1, s2), p) == subst(s2, subst(s1, p))
    True
    >>> subst(subst_compose(s2, s1), p) == subst(s1, subst(s2, p))
    True
    """
    sc = {}
    for x, v in s1.items():
        if s2.has_key(v):
            w = s2[v]
            sc[x] = w # x -> v -> w
        else:
            sc[x] = v
    for x, v in s2.items():
        if not (s1.has_key(x)):
            sc[x] = v
        # otherwise s1[x] preemptys s2[x]
    return sc

#______________________________________________________________________________

# Example application (not in the book).
# You can use the Expr class to do symbolic differentiation.  This used to be
# a part of AI; now it is considered a separate field, Symbolic Algebra.

def diff(y, x):
    """Return the symbolic derivative, dy/dx, as an Expr.
    However, you probably want to simplify the results with simp.
    >>> diff(x * x, x)
    ((x * 1) + (x * 1))
    >>> simp(diff(x * x, x))
    (2 * x)
    """
    if y == x: return ONE
    elif not y.args: return ZERO
    else:
        u, op, v = y.args[0], y.op, y.args[-1]
        if op == '+': return diff(u, x) + diff(v, x)
        elif op == '-' and len(args) == 1: return -diff(u, x)
        elif op == '-': return diff(u, x) - diff(v, x)
        elif op == '*': return u * diff(v, x) + v * diff(u, x)
        elif op == '/': return (v*diff(u, x) - u*diff(v, x)) / (v * v)
        elif op == '**' and isnumber(x.op):
            return (v * u ** (v - 1) * diff(u, x))
        elif op == '**': return (v * u ** (v - 1) * diff(u, x)
                                 + u ** v * Expr('log')(u) * diff(v, x))
        elif op == 'log': return diff(u, x) / u
        else: raise ValueError("Unknown op: %s in diff(%s, %s)" % (op, y, x))

def simp(x):
    if not x.args: return x
    args = map(simp, x.args)
    u, op, v = args[0], x.op, args[-1]
    if op == '+': 
        if v == ZERO: return u
        if u == ZERO: return v
        if u == v: return TWO * u
        if u == -v or v == -u: return ZERO
    elif op == '-' and len(args) == 1: 
        if u.op == '-' and len(u.args) == 1: return u.args[0] ## --y ==> y
    elif op == '-': 
        if v == ZERO: return u
        if u == ZERO: return -v
        if u == v: return ZERO
        if u == -v or v == -u: return ZERO
    elif op == '*': 
        if u == ZERO or v == ZERO: return ZERO
        if u == ONE: return v
        if v == ONE: return u
        if u == v: return u ** 2
    elif op == '/': 
        if u == ZERO: return ZERO
        if v == ZERO: return Expr('Undefined')
        if u == v: return ONE
        if u == -v or v == -u: return ZERO
    elif op == '**': 
        if u == ZERO: return ZERO
        if v == ZERO: return ONE
        if u == ONE: return ONE
        if v == ONE: return u
    elif op == 'log': 
        if u == ONE: return ZERO
    else: raise ValueError("Unknown op: " + op)
    ## If we fall through to here, we can not simplify further
    return Expr(op, *args)

def d(y, x):
    "Differentiate and then simplify."
    return simp(diff(y, x))    

#_______________________________________________________________________________

# Utilities for doctest cases
# These functions print their arguments in a standard order
# to compensate for the random order in the standard representation

def pretty(x):
    t = type(x)
    if t == dict:
        return pretty_dict(x)
    elif t == set:
        return pretty_set(x)

def pretty_dict(d):
    """Print the dictionary d.
    
    Prints a string representation of the dictionary
    with keys in sorted order according to their string
    representation: {a: A, d: D, ...}.
    >>> pretty_dict({'m': 'M', 'a': 'A', 'r': 'R', 'k': 'K'})
    "{'a': 'A', 'k': 'K', 'm': 'M', 'r': 'R'}"
    >>> pretty_dict({z: C, y: B, x: A})
    '{x: A, y: B, z: C}'
    """

    def format(k, v):
        return "%s: %s" % (repr(k), repr(v))

    ditems = d.items()
    ditems.sort(key=str)
    k, v = ditems[0]
    dpairs = format(k, v)
    for (k, v) in ditems[1:]:
        dpairs += (', ' + format(k, v))
    return '{%s}' % dpairs

def pretty_set(s):
    """Print the set s.

    >>> pretty_set(set(['A', 'Q', 'F', 'K', 'Y', 'B']))
    "set(['A', 'B', 'F', 'K', 'Q', 'Y'])"
    >>> pretty_set(set([z, y, x]))
    'set([x, y, z])'
    """

    slist = list(s)
    slist.sort(key=str)
    return 'set(%s)' % slist

def pp(x):
    print pretty(x)
    
def ppsubst(s):
    """Pretty-print substitution s"""
    ppdict(s)

def ppdict(d):
    print pretty_dict(d)

def ppset(s):
    print pretty_set(s)

#________________________________________________________________________

class logicTest: """
### PropKB
>>> kb = PropKB()
>>> kb.tell(A & B)
>>> kb.tell(B >> C)
>>> kb.ask(C) ## The result {} means true, with no substitutions
{}
>>> kb.ask(P) 
False
>>> kb.retract(B)
>>> kb.ask(C) 
False

>>> pl_true(P, {}) 
>>> pl_true(P | Q, {P: True}) 
True

# Notice that the function pl_true cannot reason by cases:
>>> pl_true(P | ~P) 

# However, tt_true can:
>>> tt_true(P | ~P) 
True

# The following are tautologies from [Fig. 7.11]:
>>> tt_true("(A & B) <=> (B & A)") 
True
>>> tt_true("(A | B) <=> (B | A)") 
True
>>> tt_true("((A & B) & C) <=> (A & (B & C))") 
True
>>> tt_true("((A | B) | C) <=> (A | (B | C))") 
True
>>> tt_true("~~A <=> A") 
True
>>> tt_true("(A >> B) <=> (~B >> ~A)") 
True
>>> tt_true("(A >> B) <=> (~A | B)") 
True
>>> tt_true("(A <=> B) <=> ((A >> B) & (B >> A))") 
True
>>> tt_true("~(A & B) <=> (~A | ~B)") 
True
>>> tt_true("~(A | B) <=> (~A & ~B)") 
True
>>> tt_true("(A & (B | C)) <=> ((A & B) | (A & C))") 
True
>>> tt_true("(A | (B & C)) <=> ((A | B) & (A | C))") 
True

# The following are not tautologies:
>>> tt_true(A & ~A) 
False
>>> tt_true(A & B) 
False

### [Fig. 7.13]
>>> alpha = expr("~P12")
>>> to_cnf(Fig[7,13] & ~alpha)
((~P12 | B11) & (~P21 | B11) & (P12 | P21 | ~B11) & ~B11 & P12)
>>> tt_entails(Fig[7,13], alpha) 
True
>>> pl_resolution(PropKB(Fig[7,13]), alpha) 
True

### [Fig. 7.15]
>>> pl_fc_entails(Fig[7,15], expr('SomethingSilly')) 
False

### Unification:
>>> unify(x, x, {}) 
{}
>>> unify(x, 3, {}) 
{x: 3}


>>> to_cnf((P&Q) | (~P & ~Q)) 
((~P | P) & (~Q | P) & (~P | Q) & (~Q | Q))
"""
