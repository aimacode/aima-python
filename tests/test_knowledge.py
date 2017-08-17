from knowledge import *
from utils import expr
import random

random.seed("aima-python")


def test_current_best_learning():
    examples = restaurant
    hypothesis = [{'Alt': 'Yes'}]
    h = current_best_learning(examples, hypothesis)
    values = []
    for e in examples:
        values.append(guess_value(e, h))

    assert values == [True, False, True, True, False, True, False, True, False, False, False, True]

    examples = animals_umbrellas
    initial_h = [{'Species': 'Cat'}]
    h = current_best_learning(examples, initial_h)
    values = []
    for e in examples:
        values.append(guess_value(e, h))

    assert values == [True, True, True, False, False, False, True]

    examples = party
    initial_h = [{'Pizza': 'Yes'}]
    h = current_best_learning(examples, initial_h)
    values = []
    for e in examples:
        values.append(guess_value(e, h))

    assert values == [True, True, False]


def test_version_space_learning():
    V = version_space_learning(party)
    results = []
    for e in party:
        guess = False
        for h in V:
            if guess_value(e, h):
                guess = True
                break

        results.append(guess)

    assert results == [True, True, False]
    assert [{'Pizza': 'Yes'}] in V


def test_minimal_consistent_det():
    assert minimal_consistent_det(party, {'Pizza', 'Soda'}) == {'Pizza'}
    assert minimal_consistent_det(party[:2], {'Pizza', 'Soda'}) == set()
    assert minimal_consistent_det(animals_umbrellas, {'Species', 'Rain', 'Coat'}) == {'Species', 'Rain', 'Coat'}
    assert minimal_consistent_det(conductance, {'Mass', 'Temp', 'Material', 'Size'}) == {'Temp', 'Material'}
    assert minimal_consistent_det(conductance, {'Mass', 'Temp', 'Size'}) == {'Mass', 'Temp', 'Size'}


def test_extend_example():
    assert list(test_network.extend_example({x: A, y: B}, expr('Conn(x, z)'))) == [
        {x: A, y: B, z: B}, {x: A, y: B, z: D}]
    assert list(test_network.extend_example({x: G}, expr('Conn(x, y)'))) == [{x: G, y: I}]
    assert list(test_network.extend_example({x: C}, expr('Conn(x, y)'))) == []
    assert len(list(test_network.extend_example({}, expr('Conn(x, y)')))) == 10
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Father(x, y)')))) == 2
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Mother(x, y)')))) == 0
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Female(y)')))) == 6


def test_new_literals():
    assert len(list(test_network.new_literals([expr('p | q'), [expr('p')]]))) == 8
    assert len(list(test_network.new_literals([expr('p'), [expr('q'), expr('p | r')]]))) == 15
    assert len(list(small_family.new_literals([expr('p'), []]))) == 8
    assert len(list(small_family.new_literals([expr('p & q'), []]))) == 20


def test_choose_literal():
    literals = [expr('Conn(p, q)'), expr('Conn(x, z)'), expr('Conn(r, s)'), expr('Conn(t, y)')]
    examples_pos = [{x: A, y: B}, {x: A, y: D}]
    examples_neg = [{x: A, y: C}, {x: C, y: A}, {x: C, y: B}, {x: A, y: I}]
    assert test_network.choose_literal(literals, [examples_pos, examples_neg]) == expr('Conn(x, z)')
    literals = [expr('Conn(x, p)'), expr('Conn(p, x)'), expr('Conn(p, q)')]
    examples_pos = [{x: C}, {x: F}, {x: I}]
    examples_neg = [{x: D}, {x: A}, {x: B}, {x: G}]
    assert test_network.choose_literal(literals, [examples_pos, examples_neg]) == expr('Conn(p, x)')
    literals = [expr('Father(x, y)'), expr('Father(y, x)'), expr('Mother(x, y)'), expr('Mother(x, y)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Peter')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Father(x, y)')
    literals = [expr('Father(x, y)'), expr('Father(y, x)'), expr('Male(x)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Andrew')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Male(x)')


def test_new_clause():
    target = expr('Open(x, y)')
    examples_pos = [{x: B}, {x: A}, {x: G}]
    examples_neg = [{x: C}, {x: F}, {x: I}]
    clause = test_network.new_clause([examples_pos, examples_neg], target)[0][1]
    assert len(clause) == 1 and clause[0].op == 'Conn' and clause[0].args[0] == x
    target = expr('Flow(x, y)')
    examples_pos = [{x: B}, {x: D}, {x: E}, {x: G}]
    examples_neg = [{x: A}, {x: C}, {x: F}, {x: I}, {x: H}]
    clause = test_network.new_clause([examples_pos, examples_neg], target)[0][1]
    assert len(clause) == 2 and \
        ((clause[0].args[0] == x and clause[1].args[1] == x) or \
        (clause[0].args[1] == x and clause[1].args[0] == x))


def test_foil():
    target = expr('Reach(x, y)')
    examples_pos = [{x: A, y: B},
                    {x: A, y: C},
                    {x: A, y: D},
                    {x: A, y: E},
                    {x: A, y: F},
                    {x: A, y: G},
                    {x: A, y: I},
                    {x: B, y: C},
                    {x: D, y: C},
                    {x: D, y: E},
                    {x: D, y: F},
                    {x: D, y: G},
                    {x: D, y: I},
                    {x: E, y: F},
                    {x: E, y: G},
                    {x: E, y: I},
                    {x: G, y: I},
                    {x: H, y: G},
                    {x: H, y: I}]
    nodes = {A, B, C, D, E, F, G, H, I}
    examples_neg = [example for example in [{x: a, y: b} for a in nodes for b in nodes]
                    if example not in examples_pos]
    ## TODO: Modify FOIL to recursively check for satisfied positive examples
#    clauses = test_network.foil([examples_pos, examples_neg], target)
#    assert len(clauses) == 2
    target = expr('Parent(x, y)')
    examples_pos = [{x: expr('Elizabeth'), y: expr('Anne')},
                    {x: expr('Elizabeth'), y: expr('Andrew')},
                    {x: expr('Philip'), y: expr('Anne')},
                    {x: expr('Philip'), y: expr('Andrew')},
                    {x: expr('Anne'), y: expr('Peter')},
                    {x: expr('Anne'), y: expr('Zara')},
                    {x: expr('Mark'), y: expr('Peter')},
                    {x: expr('Mark'), y: expr('Zara')},
                    {x: expr('Andrew'), y: expr('Beatrice')},
                    {x: expr('Andrew'), y: expr('Eugenie')},
                    {x: expr('Sarah'), y: expr('Beatrice')},
                    {x: expr('Sarah'), y: expr('Eugenie')}]
    examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                    {x: expr('Beatrice'), y: expr('Eugenie')},
                    {x: expr('Mark'), y: expr('Elizabeth')},
                    {x: expr('Beatrice'), y: expr('Philip')}]
    clauses = small_family.foil([examples_pos, examples_neg], target)
    assert len(clauses) == 2 and \
        ((clauses[0][1][0] == expr('Father(x, y)') and clauses[1][1][0] == expr('Mother(x, y)')) or \
        (clauses[1][1][0] == expr('Father(x, y)') and clauses[0][1][0] == expr('Mother(x, y)')))
    target = expr('Grandparent(x, y)')
    examples_pos = [{x: expr('Elizabeth'), y: expr('Peter')},
                    {x: expr('Elizabeth'), y: expr('Zara')},
                    {x: expr('Elizabeth'), y: expr('Beatrice')},
                    {x: expr('Elizabeth'), y: expr('Eugenie')},
                    {x: expr('Philip'), y: expr('Peter')},
                    {x: expr('Philip'), y: expr('Zara')},
                    {x: expr('Philip'), y: expr('Beatrice')},
                    {x: expr('Philip'), y: expr('Eugenie')}]
    examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                    {x: expr('Beatrice'), y: expr('Eugenie')},
                    {x: expr('Elizabeth'), y: expr('Andrew')},
                    {x: expr('Philip'), y: expr('Anne')},
                    {x: expr('Philip'), y: expr('Andrew')},
                    {x: expr('Anne'), y: expr('Peter')},
                    {x: expr('Anne'), y: expr('Zara')},
                    {x: expr('Mark'), y: expr('Peter')},
                    {x: expr('Mark'), y: expr('Zara')},
                    {x: expr('Andrew'), y: expr('Beatrice')},
                    {x: expr('Andrew'), y: expr('Eugenie')},
                    {x: expr('Sarah'), y: expr('Beatrice')},
                    {x: expr('Mark'), y: expr('Elizabeth')},
                    {x: expr('Beatrice'), y: expr('Philip')}]
#    clauses = small_family.foil([examples_pos, examples_neg], target)
#    assert len(clauses) == 2 and \
#        ((clauses[0][1][0] == expr('Father(x, y)') and clauses[1][1][0] == expr('Mother(x, y)')) or \
#        (clauses[1][1][0] == expr('Father(x, y)') and clauses[0][1][0] == expr('Mother(x, y)')))


party = [
    {'Pizza': 'Yes', 'Soda': 'No', 'GOAL': True},
    {'Pizza': 'Yes', 'Soda': 'Yes', 'GOAL': True},
    {'Pizza': 'No', 'Soda': 'No', 'GOAL': False}
]

animals_umbrellas = [
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': True},
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Dog', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'Yes', 'GOAL': True}
]

conductance = [
    {'Sample': 'S1', 'Mass': 12, 'Temp': 26, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.59},
    {'Sample': 'S1', 'Mass': 12, 'Temp': 100, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.57},
    {'Sample': 'S2', 'Mass': 24, 'Temp': 26, 'Material': 'Cu', 'Size': 6, 'GOAL': 0.59},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 26, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.05},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 100, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S5', 'Mass': 24, 'Temp': 100, 'Material': 'Pb', 'Size': 4, 'GOAL': 0.04},
    {'Sample': 'S6', 'Mass': 36, 'Temp': 26, 'Material': 'Pb', 'Size': 6, 'GOAL': 0.05},
]

def r_example(Alt, Bar, Fri, Hun, Pat, Price, Rain, Res, Type, Est, GOAL):
    return {'Alt': Alt, 'Bar': Bar, 'Fri': Fri, 'Hun': Hun, 'Pat': Pat,
            'Price': Price, 'Rain': Rain, 'Res': Res, 'Type': Type, 'Est': Est,
            'GOAL': GOAL}

restaurant = [
    r_example('Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', True),
    r_example('Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', False),
    r_example('No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', True),
    r_example('Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'No', 'Thai', '10-30', True),
    r_example('Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', False),
    r_example('No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', True),
    r_example('No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', False),
    r_example('No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', True),
    r_example('No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', False),
    r_example('No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', True)
]

"""
A              H
|\            /|
| \          / |
v  v        v  v
B  D-->E-->G-->I
|  /   |
| /    |
vv     v
C      F
"""
test_network = FOIL_container([expr("Conn(A, B)"),
                               expr("Conn(A ,D)"),
                               expr("Conn(B, C)"),
                               expr("Conn(D, C)"),
                               expr("Conn(D, E)"),
                               expr("Conn(E ,F)"),
                               expr("Conn(E, G)"),
                               expr("Conn(G, I)"),
                               expr("Conn(H, G)"),
                               expr("Conn(H, I)")])

small_family = FOIL_container([expr("Mother(Anne, Peter)"),
                               expr("Mother(Anne, Zara)"),
                               expr("Mother(Sarah, Beatrice)"),
                               expr("Mother(Sarah, Eugenie)"),
                               expr("Father(Mark, Peter)"),
                               expr("Father(Mark, Zara)"),
                               expr("Father(Andrew, Beatrice)"),
                               expr("Father(Andrew, Eugenie)"),
                               expr("Father(Philip, Anne)"),
                               expr("Father(Philip, Andrew)"),
                               expr("Mother(Elizabeth, Anne)"),
                               expr("Mother(Elizabeth, Andrew)"),
                               expr("Male(Philip)"),
                               expr("Male(Mark)"),
                               expr("Male(Andrew)"),
                               expr("Male(Peter)"),
                               expr("Female(Elizabeth)"),
                               expr("Female(Anne)"),
                               expr("Female(Sarah)"),
                               expr("Female(Zara)"),
                               expr("Female(Beatrice)"),
                               expr("Female(Eugenie)"),
])

A, B, C, D, E, F, G, H, I, x, y, z = map(expr, 'ABCDEFGHIxyz')
