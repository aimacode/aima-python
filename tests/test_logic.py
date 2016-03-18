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

def test_pl_fc_entails():
    assert pl_fc_entails(Fig[7,15], expr('Q'))
    assert not pl_fc_entails(Fig[7,15], expr('SomethingSilly'))

def test_tt_entails():
    assert tt_entails(P & Q, Q)
    assert not tt_entails(P | Q, Q)
    assert tt_entails(A & (B | C) & E & F & ~(P | Q), A & E & F & ~P & ~Q)

def test_eliminate_implications():
    assert repr(eliminate_implications(A >> (~B << C))) == '((~B | ~C) | ~A)'
    assert repr(eliminate_implications(A ^ B)) == '((A & ~B) | (~A & B))'
    assert repr(eliminate_implications(A & B | C & ~D)) == '((A & B) | (C & ~D))'

def test_dissociate():
    assert dissociate('&', [A & B]) == [A, B]
    assert dissociate('|', [A, B, C & D, P | Q]) == [A, B, C & D, P, Q]
    assert dissociate('&', [A, B, C & D, P | Q]) == [A, B, C, D, P | Q]

def test_associate():
    assert repr(associate('&', [(A&B),(B|C),(B&C)])) == '(A & B & (B | C) & B & C)'
    assert repr(associate('|', [A|(B|(C|(A&B)))])) == '(A | B | C | (A & B))'

def test_move_not_inwards():
    assert repr(move_not_inwards(~(A | B))) == '(~A & ~B)'
    assert repr(move_not_inwards(~(A & B))) == '(~A | ~B)'
    assert repr(move_not_inwards(~(~(A | ~B) | ~~C))) == '((A | ~B) & ~C)'

def test_to_cnf():
    assert repr(to_cnf(Fig[7, 13] & ~expr('~P12'))) == \
        "((~P12 | B11) & (~P21 | B11) & (P12 | P21 | ~B11) & ~B11 & P12)"
    assert repr(to_cnf((P&Q) | (~P & ~Q))) == '((~P | P) & (~Q | P) & (~P | Q) & (~Q | Q))'

def test_fol_bc_ask():    
    def test_ask(query, kb=None):
        q = expr(query)
        vars = variables(q)
        answers = fol_bc_ask(kb or test_kb, q)
        return sorted(
            [dict((x, v) for x, v in list(a.items()) if x in vars)
             for a in answers],  key=repr)
    assert repr(test_ask('Farmer(x)')) == '[{x: Mac}]'
    assert repr(test_ask('Human(x)')) == '[{x: Mac}, {x: MrsMac}]'
    assert repr(test_ask('Rabbit(x)')) == '[{x: MrsRabbit}, {x: Pete}]'
    assert repr(test_ask('Criminal(x)', crime_kb)) == '[{x: West}]'


if __name__ == '__main__':
    pytest.main()
