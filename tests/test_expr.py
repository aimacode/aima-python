import pytest
from expr import *

def test_Expr():
    A, B, C = symbols('A, B, C')
    assert all(isinstance(S, Symbol) for S in (A, B, C))
    assert A.name == 'A'
    assert arity(A) == 0 and A.args == ()
    
    b = Expr('+', A, 1)
    assert arity(b) == 2 and b.op == '+' and b.args == (A, 1)
    
    u = Expr('-', b)
    assert arity(u) == 1 and u.op == '-' and u.args == (b,)
    
    assert (b ** u) == (b ** u)
    assert (b ** u) != (1 ** A)
    
    assert A + b * C ** 2 == A + (b * (C ** 2))
    
    assert A in (C >> 1 / (A % 1))
    assert B not in (C >> 1 / (A % 1))
    
def test_ex():
    P, Q, x, y, z, GP = symbols('P Q x y z GP')
    assert ex('1 + 2 * x') == Expr('+', 1, Expr('*', 2, x))
    assert ex('~', 'P') == ex('~P') == Expr('~', Symbol('P'))
    assert ex('P & Q', '==>', 'P') == Expr('==>', P & Q, P)
    assert ex('P & Q', '<=>', 'Q & P') == Expr('<=>', (P & Q), (Q & P))
    assert ex(x + y, '==', y + x) == Expr('==', x + y, y + x)
    assert ex('P(x) | P(y) & Q(z)') == (P(x) | (P(y) & Q(z)))
    # x is grandparent of z if x is parent of y and y is parent of z:
    assert (ex('GP(x, z)', '<==', 'P(x, y) & P(y, z)')
            == Expr('<==', GP(x, z), P(x, y) & P(y, z)))


if __name__ == '__main__':
    pytest.main()
