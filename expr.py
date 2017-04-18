import re
import functools

@functools.total_ordering
class Expr(object): 
    """A mathematical expression with an operator and 1 or more arguments.
    op is a str like '+' or 'sin'; args are either Exprs or numbers.
    Expr('-', x) is a unary expression; Expr('+', x, 1) a binary."""
    
    def __init__(self, op, *args): 
        if not isinstance(op, str):
            raise TypeError('Expr op should be a string, not {}'.format(op))
        for arg in args:
            if not isinstance(arg, Expression):
                raise TypeError('Expr arg should be an Expression, not {}'.format(arg))
        self.op = op
        self.args = args
        
    # Operator overloads
    def __add__(self, other): return Expr('+',  self, other)
    def __sub__(self, other): return Expr('-',  self, other)
    def __mul__(self, other): return Expr('*',  self, other)
    def __pow__(self, other): return Expr('**', self, other)
    def __mod__(self, other): return Expr('%',  self, other)
    def __and__(self, other): return Expr('&',  self, other)
    def __xor__(self, other): return Expr('^',  self, other)
    def __or__(self, other):  return Expr('|',  self, other)
    def __rshift__(self, other):   return Expr('>>',  self, other)
    def __lshift__(self, other):   return Expr('<<',  self, other)
    def __truediv__(self, other):  return Expr('/',  self, other)
    def __floordiv__(self, other): return Expr('//',  self, other)
    
    # Reverse operator overloads
    def __radd__(self, other): return Expr('+',  other, self)
    def __rsub__(self, other): return Expr('-',  other, self)
    def __rmul__(self, other): return Expr('*',  other, self)
    def __rdiv__(self, other): return Expr('/',  other, self)
    def __rpow__(self, other): return Expr('**', other, self)
    def __rmod__(self, other): return Expr('%',  other, self)
    def __rand__(self, other): return Expr('&',  other, self)
    def __rxor__(self, other): return Expr('^',  other, self)
    def __ror__(self, other):  return Expr('|',  other, self)
    def __rrshift__(self, other):   return Expr('>>',  other, self)
    def __rlshift__(self, other):   return Expr('<<',  other, self)
    def __rtruediv__(self, other):  return Expr('/',  other, self)
    def __rfloordiv__(self, other): return Expr('//',  other, self)
    
    # Unary operator overloads
    def __neg__(self):    return Expr('-', self)
    def __pos__(self):    return Expr('+', self)
    def __invert__(self): return Expr('~', self)  
    
    def __call__(self, *args): 
        "Call: for example, use f(x) to create Expr('f', x)"
        return Expr(self.op, *args)
    
    def __contains__(self, x): 
        "'x in exp' tests if x == exp or x in exp.args."
        return (x == self 
                or x in self.args 
                or any(isinstance(a, Expr) and x in a
                       for a in self.args))
     
    def __eq__(self, other):   
        "'x == y' evaluates to True or False; does not build an Expr."
        return (isinstance(other, Expr) 
                and self.op == other.op 
                and self.args == other.args)
    
    def __lt__(self, other):   
        "'x <= y' evaluates to True or False; does not build an Expr."
        return (isinstance(other, Expr) 
                and self.op < other.op 
                and self.args < other.args)
    
    def __hash__(self): return hash(self.op) ^ hash(self.args)
    
    def __repr__(self):
        op   = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():       # f(x) or f(x, y)
            return '{}({})'.format(op, ', '.join(args))
        elif len(args) == 1:        # -x or -(x + 1)
            return op + args[0]
        else:                       # (x - y)
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'

class Symbol(Expr):
    "An atomic Expr, such as: x = Symbol('x')"
    
    def __init__(self, name):
        self.name = name
        Expr.__init__(self, name) # Symbols are Exprs
        
    def __repr__(self): return self.name

def symbols(names):
    "Return a list of Symbols; names is a comma/whitespace delimited str."
    return [Symbol(name) for name in names.replace(',', ' ').split()]

# An 'Expression' is either an 'Expr' or a number:
Expression = (Expr, int, float, complex)

def arity(expression):
    "The number of sub-expressions in this expression."
    if isinstance(expression, Expr):
        return len(expression.args)
    else: # expression is a number
        return 0
                
def ex(arg1, arg2=None, arg3=None):
    """Shortcut to create an Expr, automatically defining symbols. 
    With 1 arg, arg can be an Expr or a string denoting one.
    With 2 args, arg1 is a unary op; with 3 args, arg2 is a binary op.
    The other (non-operator) args can be strings or Exprs, for example:
    >>> expr('P(x) | P(y) & Q(z)')    # 1 args
    (P(x) | (P(y) & Q(z)))
    >>> expr('~', 'P')                # 2 args
    ~P
    >>> expr('P & Q', '<=>', 'Q & P') # 3 args
    ((P & Q) <=> (Q & P))
    """
    if arg2 is None and arg3 is None: # 1 arg
        if isinstance(arg1, Expression): 
            return arg1
        else:
            syms = {var: Symbol(var) for var in re.findall(r'\w+', arg1)}
            return eval(arg1, syms)
    elif arg3 is None:               # 2 args
        return Expr(arg1, ex(arg2))
    else:                            # 3 args
        return Expr(arg2, ex(arg1), ex(arg3))