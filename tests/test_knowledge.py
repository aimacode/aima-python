import pytest

from knowledge import *
from utils import expr
import random

random.seed("aima-python")

party = [
    {'Pizza': 'Yes', 'Soda': 'No', 'GOAL': True},
    {'Pizza': 'Yes', 'Soda': 'Yes', 'GOAL': True},
    {'Pizza': 'No', 'Soda': 'No', 'GOAL': False}]

animals_umbrellas = [
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': True},
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Dog', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'Yes', 'GOAL': True}]

conductance = [
    {'Sample': 'S1', 'Mass': 12, 'Temp': 26, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.59},
    {'Sample': 'S1', 'Mass': 12, 'Temp': 100, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.57},
    {'Sample': 'S2', 'Mass': 24, 'Temp': 26, 'Material': 'Cu', 'Size': 6, 'GOAL': 0.59},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 26, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.05},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 100, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S5', 'Mass': 24, 'Temp': 100, 'Material': 'Pb', 'Size': 4, 'GOAL': 0.04},
    {'Sample': 'S6', 'Mass': 36, 'Temp': 26, 'Material': 'Pb', 'Size': 6, 'GOAL': 0.05}]


def r_example(Alt, Bar, Fri, Hun, Pat, Price, Rain, Res, Type, Est, GOAL):
    return {'Alt': Alt, 'Bar': Bar, 'Fri': Fri, 'Hun': Hun, 'Pat': Pat, 'Price': Price,
            'Rain': Rain, 'Res': Res, 'Type': Type, 'Est': Est, 'GOAL': GOAL}


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
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', True)]


def test_current_best_learning():
    examples = restaurant
    hypothesis = [{'Alt': 'Yes'}]
    h = current_best_learning(examples, hypothesis)
    values = [guess_value(e, h) for e in examples]

    assert values == [True, False, True, True, False, True, False, True, False, False, False, True]

    examples = animals_umbrellas
    initial_h = [{'Species': 'Cat'}]
    h = current_best_learning(examples, initial_h)
    values = [guess_value(e, h) for e in examples]

    assert values == [True, True, True, False, False, False, True]

    examples = party
    initial_h = [{'Pizza': 'Yes'}]
    h = current_best_learning(examples, initial_h)
    values = [guess_value(e, h) for e in examples]

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


A, B, C, D, E, F, G, H, I, x, y, z = map(expr, 'ABCDEFGHIxyz')

# knowledge base containing family relations
small_family = FOILContainer([expr("Mother(Anne, Peter)"),
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
                              expr("Female(Eugenie)")])

smaller_family = FOILContainer([expr("Mother(Anne, Peter)"),
                                expr("Father(Mark, Peter)"),
                                expr("Father(Philip, Anne)"),
                                expr("Mother(Elizabeth, Anne)"),
                                expr("Male(Philip)"),
                                expr("Male(Mark)"),
                                expr("Male(Peter)"),
                                expr("Female(Elizabeth)"),
                                expr("Female(Anne)")])

# target relation
target = expr('Parent(x, y)')

# positive examples of target
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

# negative examples of target
examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                {x: expr('Beatrice'), y: expr('Eugenie')},
                {x: expr('Mark'), y: expr('Elizabeth')},
                {x: expr('Beatrice'), y: expr('Philip')}]


def test_tell():
    """
    adds in the knowledge base a sentence
    """
    smaller_family.tell(expr("Male(George)"))
    smaller_family.tell(expr("Female(Mum)"))
    assert smaller_family.ask(expr("Male(George)")) == {}
    assert smaller_family.ask(expr("Female(Mum)")) == {}
    assert not smaller_family.ask(expr("Female(George)"))
    assert not smaller_family.ask(expr("Male(Mum)"))


def test_extend_example():
    """
    Create the extended examples of the given clause. 
    (The extended examples are a set of examples created by extending example 
    with each possible constant value for each new variable in literal.)
    """
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Father(x, y)')))) == 2
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Mother(x, y)')))) == 0
    assert len(list(small_family.extend_example({x: expr('Andrew')}, expr('Female(y)')))) == 6


def test_new_literals():
    assert len(list(small_family.new_literals([expr('p'), []]))) == 8
    assert len(list(small_family.new_literals([expr('p & q'), []]))) == 20


def test_new_clause():
    """
    Finds the best clause to add in the set of clauses.
    """
    clause = small_family.new_clause([examples_pos, examples_neg], target)[0][1]
    assert len(clause) == 1 and (clause[0].op in ['Male', 'Female', 'Father', 'Mother'])


def test_choose_literal():
    """
    Choose the best literal based on the information gain
    """
    literals = [expr('Father(x, y)'), expr('Father(x, y)'), expr('Mother(x, y)'), expr('Mother(x, y)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Peter')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Father(x, y)')
    literals = [expr('Father(x, y)'), expr('Father(y, x)'), expr('Male(x)')]
    examples_pos = [{x: expr('Philip')}, {x: expr('Mark')}, {x: expr('Andrew')}]
    examples_neg = [{x: expr('Elizabeth')}, {x: expr('Sarah')}]
    assert small_family.choose_literal(literals, [examples_pos, examples_neg]) == expr('Father(x,y)')


def test_gain():
    """
    Calculates the utility of each literal, based on the information gained. 
    """
    gain_father = small_family.gain(expr('Father(x,y)'), [examples_pos, examples_neg])
    gain_male = small_family.gain(expr('Male(x)'), [examples_pos, examples_neg])
    assert round(gain_father, 2) == 2.49
    assert round(gain_male, 2) == 1.16


def test_update_examples():
    """Add to the kb those examples what are represented in extended_examples
        List of omitted examples is returned.
    """
    extended_examples = [{x: expr("Mark"), y: expr("Peter")},
                         {x: expr("Philip"), y: expr("Anne")}]

    uncovered = smaller_family.update_examples(target, examples_pos, extended_examples)
    assert {x: expr("Elizabeth"), y: expr("Anne")} in uncovered
    assert {x: expr("Anne"), y: expr("Peter")} in uncovered
    assert {x: expr("Philip"), y: expr("Anne")} not in uncovered
    assert {x: expr("Mark"), y: expr("Peter")} not in uncovered


def test_foil():
    """
    Test the FOIL algorithm, when target is  Parent(x,y)
    """
    clauses = small_family.foil([examples_pos, examples_neg], target)
    assert len(clauses) == 2 and \
           ((clauses[0][1][0] == expr('Father(x, y)') and clauses[1][1][0] == expr('Mother(x, y)')) or
            (clauses[1][1][0] == expr('Father(x, y)') and clauses[0][1][0] == expr('Mother(x, y)')))

    target_g = expr('Grandparent(x, y)')
    examples_pos_g = [{x: expr('Elizabeth'), y: expr('Peter')},
                      {x: expr('Elizabeth'), y: expr('Zara')},
                      {x: expr('Elizabeth'), y: expr('Beatrice')},
                      {x: expr('Elizabeth'), y: expr('Eugenie')},
                      {x: expr('Philip'), y: expr('Peter')},
                      {x: expr('Philip'), y: expr('Zara')},
                      {x: expr('Philip'), y: expr('Beatrice')},
                      {x: expr('Philip'), y: expr('Eugenie')}]
    examples_neg_g = [{x: expr('Anne'), y: expr('Eugenie')},
                      {x: expr('Beatrice'), y: expr('Eugenie')},
                      {x: expr('Elizabeth'), y: expr('Andrew')},
                      {x: expr('Elizabeth'), y: expr('Anne')},
                      {x: expr('Elizabeth'), y: expr('Mark')},
                      {x: expr('Elizabeth'), y: expr('Sarah')},
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
                      {x: expr('Beatrice'), y: expr('Philip')},
                      {x: expr('Peter'), y: expr('Andrew')},
                      {x: expr('Zara'), y: expr('Mark')},
                      {x: expr('Peter'), y: expr('Anne')},
                      {x: expr('Zara'), y: expr('Eugenie')}]

    clauses = small_family.foil([examples_pos_g, examples_neg_g], target_g)
    assert len(clauses[0]) == 2
    assert clauses[0][1][0].op == 'Parent'
    assert clauses[0][1][0].args[0] == x
    assert clauses[0][1][1].op == 'Parent'
    assert clauses[0][1][1].args[1] == y


if __name__ == "__main__":
    pytest.main()
