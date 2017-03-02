import pytest
from logic import *
from utils import expr_handle_infix_ops, count


def test_expr():
    assert repr(expr('P <=> Q(1)')) == '(P <=> Q(1))'
    assert repr(expr('P & Q | ~R(x, F(x))')) == '((P & Q) | ~R(x, F(x)))'
    assert (expr_handle_infix_ops('P & Q ==> R & ~S')
            == "P & Q |'==>'| R & ~S")


def test_extend():
    assert extend({x: 1}, y, 2) == {x: 1, y: 2}


def test_PropKB():
    kb = PropKB()
    assert count(kb.ask(expr) for expr in [A, C, D, E, Q]) is 0
    kb.tell(A & E)
    assert kb.ask(A) == kb.ask(E) == {}
    kb.tell(E |'==>'| C)
    assert kb.ask(C) == {}
    kb.retract(E)
    assert kb.ask(E) is False
    assert kb.ask(C) is False


def test_KB_wumpus():
    # A simple KB that defines the relevant conditions of the Wumpus World as in Fig 7.4.
    # See Sec. 7.4.3
    kb_wumpus = PropKB()

    # Creating the relevant expressions
    # TODO: Let's just use P11, P12, ... = symbols('P11, P12, ...')
    P = {}
    B = {}
    P[1, 1] = Symbol("P[1,1]")
    P[1, 2] = Symbol("P[1,2]")
    P[2, 1] = Symbol("P[2,1]")
    P[2, 2] = Symbol("P[2,2]")
    P[3, 1] = Symbol("P[3,1]")
    B[1, 1] = Symbol("B[1,1]")
    B[2, 1] = Symbol("B[2,1]")

    kb_wumpus.tell(~P[1, 1])
    kb_wumpus.tell(B[1, 1] | '<=>' | ((P[1, 2] | P[2, 1])))
    kb_wumpus.tell(B[2, 1] | '<=>' | ((P[1, 1] | P[2, 2] | P[3, 1])))
    kb_wumpus.tell(~B[1, 1])
    kb_wumpus.tell(B[2, 1])

    # Statement: There is no pit in [1,1].
    assert kb_wumpus.ask(~P[1, 1]) == {}

    # Statement: There is no pit in [1,2].
    assert kb_wumpus.ask(~P[1, 2]) == {}

    # Statement: There is a pit in [2,2].
    assert kb_wumpus.ask(P[2, 2]) == False

    # Statement: There is a pit in [3,1].
    assert kb_wumpus.ask(P[3, 1]) == False

    # Statement: Neither [1,2] nor [2,1] contains a pit.
    assert kb_wumpus.ask(~P[1, 2] & ~P[2, 1]) == {}

    # Statement: There is a pit in either [2,2] or [3,1].
    assert kb_wumpus.ask(P[2, 2] | P[3, 1]) == {}


def test_definite_clause():
    assert is_definite_clause(expr('A & B & C & D ==> E'))
    assert is_definite_clause(expr('Farmer(Mac)'))
    assert not is_definite_clause(expr('~Farmer(Mac)'))
    assert is_definite_clause(expr('(Farmer(f) & Rabbit(r)) ==> Hates(f, r)'))
    assert not is_definite_clause(expr('(Farmer(f) & ~Rabbit(r)) ==> Hates(f, r)'))
    assert not is_definite_clause(expr('(Farmer(f) | Rabbit(r)) ==> Hates(f, r)'))


def test_pl_true():
    assert pl_true(P, {}) is None
    assert pl_true(P, {P: False}) is False
    assert pl_true(P | Q, {P: True}) is True
    assert pl_true((A | B) & (C | D), {A: False, B: True, D: True}) is True
    assert pl_true((A & B) & (C | D), {A: False, B: True, D: True}) is False
    assert pl_true((A & B) | (A & C), {A: False, B: True, C: True}) is False
    assert pl_true((A | B) & (C | D), {A: True, D: False}) is None
    assert pl_true(P | P, {}) is None


def test_tt_true():
    assert tt_true(P | ~P)
    assert tt_true('~~P <=> P')
    assert not tt_true((P | ~Q) & (~P | Q))
    assert not tt_true(P & ~P)
    assert not tt_true(P & Q)
    assert tt_true((P | ~Q) | (~P | Q))
    assert tt_true('(A & B) ==> (A | B)')
    assert tt_true('((A & B) & C) <=> (A & (B & C))')
    assert tt_true('((A | B) | C) <=> (A | (B | C))')
    assert tt_true('(A ==> B) <=> (~B ==> ~A)')
    assert tt_true('(A ==> B) <=> (~A | B)')
    assert tt_true('(A <=> B) <=> ((A ==> B) & (B ==> A))')
    assert tt_true('~(A & B) <=> (~A | ~B)')
    assert tt_true('~(A | B) <=> (~A & ~B)')
    assert tt_true('(A & (B | C)) <=> ((A & B) | (A & C))')
    assert tt_true('(A | (B & C)) <=> ((A | B) & (A | C))')


def test_dpll():
    assert (dpll_satisfiable(A & ~B & C & (A | ~D) & (~E | ~D) & (C | ~D) & (~A | ~F) & (E | ~F)
                             & (~D | ~F) & (B | ~C | D) & (A | ~E | F) & (~A | E | D))
            == {B: False, C: True, A: True, F: False, D: True, E: False})
    assert dpll_satisfiable(A & ~B) == {A: True, B: False}
    assert dpll_satisfiable(P & ~P) == False


def test_unify():
    assert unify(x, x, {}) == {}
    assert unify(x, 3, {}) == {x: 3}


def test_pl_fc_entails():
    assert pl_fc_entails(horn_clauses_KB, expr('Q'))
    assert not pl_fc_entails(horn_clauses_KB, expr('SomethingSilly'))


def test_tt_entails():
    assert tt_entails(P & Q, Q)
    assert not tt_entails(P | Q, Q)
    assert tt_entails(A & (B | C) & E & F & ~(P | Q), A & E & F & ~P & ~Q)


def test_eliminate_implications():
    assert repr(eliminate_implications('A ==> (~B <== C)')) == '((~B | ~C) | ~A)'
    assert repr(eliminate_implications(A ^ B)) == '((A & ~B) | (~A & B))'
    assert repr(eliminate_implications(A & B | C & ~D)) == '((A & B) | (C & ~D))'


def test_dissociate():
    assert dissociate('&', [A & B]) == [A, B]
    assert dissociate('|', [A, B, C & D, P | Q]) == [A, B, C & D, P, Q]
    assert dissociate('&', [A, B, C & D, P | Q]) == [A, B, C, D, P | Q]


def test_associate():
    assert (repr(associate('&', [(A & B), (B | C), (B & C)]))
            == '(A & B & (B | C) & B & C)')
    assert (repr(associate('|', [A | (B | (C | (A & B)))]))
            == '(A | B | C | (A & B))')


def test_move_not_inwards():
    assert repr(move_not_inwards(~(A | B))) == '(~A & ~B)'
    assert repr(move_not_inwards(~(A & B))) == '(~A | ~B)'
    assert repr(move_not_inwards(~(~(A | ~B) | ~~C))) == '((A | ~B) & ~C)'


def test_to_cnf():
    assert (repr(to_cnf(wumpus_world_inference & ~expr('~P12'))) ==
            "((~P12 | B11) & (~P21 | B11) & (P12 | P21 | ~B11) & ~B11 & P12)")
    assert repr(to_cnf((P&Q) | (~P & ~Q))) == '((~P | P) & (~Q | P) & (~P | Q) & (~Q | Q))'
    assert repr(to_cnf("B <=> (P1 | P2)")) == '((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))'
    assert repr(to_cnf("a | (b & c) | d")) == '((b | a | d) & (c | a | d))'
    assert repr(to_cnf("A & (B | (D & E))")) == '(A & (D | B) & (E | B))'
    assert repr(to_cnf("A | (B | (C | (D & E)))")) == '((D | A | B | C) & (E | A | B | C))'


def test_standardize_variables():
    e = expr('F(a, b, c) & G(c, A, 23)')
    assert len(variables(standardize_variables(e))) == 3
    #assert variables(e).intersection(variables(standardize_variables(e))) == {}
    assert is_variable(standardize_variables(expr('x')))


def test_fol_bc_ask():
    def test_ask(query, kb=None):
        q = expr(query)
        test_variables = variables(q)
        answers = fol_bc_ask(kb or test_kb, q)
        return sorted(
            [dict((x, v) for x, v in list(a.items()) if x in test_variables)
             for a in answers], key=repr)
    assert repr(test_ask('Farmer(x)')) == '[{x: Mac}]'
    assert repr(test_ask('Human(x)')) == '[{x: Mac}, {x: MrsMac}]'
    assert repr(test_ask('Rabbit(x)')) == '[{x: MrsRabbit}, {x: Pete}]'
    assert repr(test_ask('Criminal(x)', crime_kb)) == '[{x: West}]'


def test_d():
    assert d(x * x - x, x) == 2 * x - 1


def test_WalkSAT():
    def check_SAT(clauses, single_solution={}):
        # Make sure the solution is correct if it is returned by WalkSat
        # Sometimes WalkSat may run out of flips before finding a solution
        soln = WalkSAT(clauses)
        if soln:
            assert all(pl_true(x, soln) for x in clauses)
            if single_solution:  # Cross check the solution if only one exists
                assert all(pl_true(x, single_solution) for x in clauses)
                assert soln == single_solution
    # Test WalkSat for problems with solution
    check_SAT([A & B, A & C])
    check_SAT([A | B, P & Q, P & B])
    check_SAT([A & B, C | D, ~(D | P)], {A: True, B: True, C: True, D: False, P: False})
    # Test WalkSat for problems without solution
    assert WalkSAT([A & ~A], 0.5, 100) is None
    assert WalkSAT([A | B, ~A, ~(B | C), C | D, P | Q], 0.5, 100) is None
    assert WalkSAT([A | B, B & C, C | D, D & A, P, ~P], 0.5, 100) is None


def test_SAT_plan():
    transition = {'A': {'Left': 'A', 'Right': 'B'},
                  'B': {'Left': 'A', 'Right': 'C'},
                  'C': {'Left': 'B', 'Right': 'C'}}
    assert SAT_plan('A', transition, 'C', 2) is None
    assert SAT_plan('A', transition, 'B', 3) == ['Right']
    assert SAT_plan('C', transition, 'A', 3) == ['Left', 'Left']

    transition = {(0, 0): {'Right': (0, 1), 'Down': (1, 0)},
                  (0, 1): {'Left': (1, 0), 'Down': (1, 1)},
                  (1, 0): {'Right': (1, 0), 'Up': (1, 0), 'Left': (1, 0), 'Down': (1, 0)},
                  (1, 1): {'Left': (1, 0), 'Up': (0, 1)}}
    assert SAT_plan((0, 0), transition, (1, 1), 4) == ['Right', 'Down']


if __name__ == '__main__':
    pytest.main()
