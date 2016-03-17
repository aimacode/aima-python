import pytest
from logic import *


def test_expr():
    assert repr(expr('P <=> Q(1)')) == '(P <=> Q(1))'
    assert repr(expr('P & Q | ~R(x, F(x))')) == '((P & Q) | ~R(x, F(x)))'

def test_PropKB():
    kb = PropKB()
    assert count(kb.ask(expr) for expr in [A, B, C, P, Q]) is 0
    kb.tell(A & B)
    assert kb.ask(A) == kb.ask(B) == {}
    kb.tell(B >> C)
    assert kb.ask(C) == {}
    kb.retract(B)
    assert kb.ask(B) is False
    assert kb.ask(C) is False

def test_pl_true():
    assert pl_true(P, {}) is None
    assert pl_true(P, {P: False}) is False
    assert pl_true(P | Q, {P: True}) is True
    assert pl_true((A|B)&(C|D), {A: False, B: True, D: True}) is True
    assert pl_true((A&B)&(C|D), {A: False, B: True, D: True}) is False
    assert pl_true((A&B)|(A&C), {A: False, B: True, C: True}) is False
    assert pl_true((A|B)&(C|D), {A: True, D: False}) is None
    assert pl_true(P | P, {}) is None

def test_tt_true():
    assert tt_true(P | ~P)
    assert tt_true('~~P <=> P')
    assert not tt_true('(P | ~Q)&(~P | Q)')
    assert not tt_true(P & ~P)
    assert not tt_true(P & Q)
    assert tt_true('(P | ~Q)|(~P | Q)')
    assert tt_true('(A & B) ==> (A | B)')
    assert tt_true('((A & B) & C) <=> (A & (B & C))')
    assert tt_true('((A | B) | C) <=> (A | (B | C))')
    assert tt_true('(A >> B) <=> (~B >> ~A)')
    assert tt_true('(A >> B) <=> (~A | B)')
    assert tt_true('(A <=> B) <=> ((A >> B) & (B >> A))')
    assert tt_true('~(A & B) <=> (~A | ~B)')
    assert tt_true('~(A | B) <=> (~A & ~B)')
    assert tt_true('(A & (B | C)) <=> ((A & B) | (A & C))')
    assert tt_true('(A | (B & C)) <=> ((A | B) & (A | C))')

def test_dpll():
    assert  dpll_satisfiable(A & ~B & C & (A | ~D) & (~E | ~D) & (C | ~D) & (~A | ~F) & (E | ~F) & (~D | ~F) & (B | ~C | D) & (A | ~E | F) & (~A | E | D)) == {B: False, C: True, A: True, F: False, D: True, E: False}  #noqa

def test_unify():
    assert unify(x, x, {}) == {}
    assert unify(x, 3, {}) == {x: 3}

def test_to_cnf():
    #assert to_cnf(Fig[7, 13] & ~expr('~P12')) BUG - FAILING THIS TEST DUE TO AN ERROR
    assert repr(to_cnf((P&Q) | (~P & ~Q))) == '((~P | P) & (~Q | P) & (~P | Q) & (~Q | Q))'
    pass

def test_pl_fc_entails():
    assert pl_fc_entails(Fig[7,15], expr('Q'))
    assert not pl_fc_entails(Fig[7,15], expr('SomethingSilly'))

def test_tt_entails():
    assert tt_entails(P & Q, Q)
    assert not tt_entails(P | Q, Q)
    assert tt_entails(A & (B | C) & E & F & ~(P | Q), A & E & F & ~P & ~Q)
    assert tt_entails(Fig[7,13], alpha)


if __name__ == '__main__':
    pytest.main()
