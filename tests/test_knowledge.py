from knowledge import *
import random

random.seed("aima-python")


# def test_current_best_learning_full():
#     """ !!! This takes too long !!! """
#     examples = restaurant
#     hypothesis = {'Alt': 'Yes'}
#     h = current_best_learning(examples, hypothesis)
#     values = []
#     for e in examples:
#         values.append(guess_value(e, h))
#
#     assert values == [True, False, True, True, False, True, False, True, False, False, False, True]


def test_current_best_learning():
    examples = restaurant[:3]
    hypothesis = {'Alt': 'Yes'}
    h = current_best_learning(examples, hypothesis)
    values = []
    for e in examples:
        values.append(guess_value(e, h))

    assert values == [True, False, True]


restaurant = [
    {'Alt': 'Yes', 'Bar': 'No', 'Fri': 'No', 'Hun': 'Yes', 'Pat': 'Some',
    'Price': '$$$', 'Rain': 'No', 'Res': 'Yes', 'Type': 'French', 'Est': '0-10',
    'GOAL': True},

    {'Alt': 'Yes', 'Bar': 'No', 'Fri': 'No', 'Hun': 'Yes', 'Pat': 'Full',
    'Price': '$', 'Rain': 'No', 'Res': 'No', 'Type': 'Thai', 'Est': '30-60',
    'GOAL': False},

    {'Alt': 'No', 'Bar': 'Yes', 'Fri': 'No', 'Hun': 'No', 'Pat': 'Some',
    'Price': '$', 'Rain': 'No', 'Res': 'No', 'Type': 'Burger', 'Est': '0-10',
    'GOAL': True},

    {'Alt': 'Yes', 'Bar': 'No', 'Fri': 'Yes', 'Hun': 'Yes', 'Pat': 'Full',
    'Price': '$', 'Rain': 'Yes', 'Res': 'No', 'Type': 'Thai', 'Est': '10-30',
    'GOAL': True},

    {'Alt': 'Yes', 'Bar': 'No', 'Fri': 'Yes', 'Hun': 'No', 'Pat': 'Full',
    'Price': '$$$', 'Rain': 'No', 'Res': 'Yes', 'Type': 'French', 'Est': '>60',
    'GOAL': False},

    {'Alt': 'No', 'Bar': 'Yes', 'Fri': 'No', 'Hun': 'Yes', 'Pat': 'Some',
    'Price': '$$', 'Rain': 'Yes', 'Res': 'Yes', 'Type': 'Italian', 'Est': '0-10',
    'GOAL': True},

    {'Alt': 'No', 'Bar': 'Yes', 'Fri': 'No', 'Hun': 'No', 'Pat': 'None',
    'Price': '$', 'Rain': 'Yes', 'Res': 'No', 'Type': 'Burger', 'Est': '0-10',
    'GOAL': False},

    {'Alt': 'No', 'Bar': 'No', 'Fri': 'No', 'Hun': 'Yes', 'Pat': 'Some',
    'Price': '$$', 'Rain': 'Yes', 'Res': 'Yes', 'Type': 'Thai', 'Est': '0-10',
    'GOAL': True},

    {'Alt': 'No', 'Bar': 'Yes', 'Fri': 'Yes', 'Hun': 'No', 'Pat': 'Full',
    'Price': '$', 'Rain': 'Yes', 'Res': 'No', 'Type': 'Burger', 'Est': '>60',
    'GOAL': False},

    {'Alt': 'Yes', 'Bar': 'Yes', 'Fri': 'Yes', 'Hun': 'Yes', 'Pat': 'Full',
    'Price': '$$$', 'Rain': 'No', 'Res': 'Yes', 'Type': 'Italian', 'Est': '10-30',
    'GOAL': False},

    {'Alt': 'No', 'Bar': 'No', 'Fri': 'No', 'Hun': 'No', 'Pat': 'None',
    'Price': '$', 'Rain': 'No', 'Res': 'No', 'Type': 'Thai', 'Est': '0-10',
    'GOAL': False},

    {'Alt': 'Yes', 'Bar': 'Yes', 'Fri': 'Yes', 'Hun': 'Yes', 'Pat': 'Full',
    'Price': '$', 'Rain': 'No', 'Res': 'No', 'Type': 'Burger', 'Est': '30-60',
    'GOAL': True}
]
