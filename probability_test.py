import pytest
from random import *
from probability import *

def tests():
    cpt = burglary.variable_node('Alarm').cpt
    parents = ['Burglary', 'Earthquake']
    event = {'Burglary': True, 'Earthquake': True}
    bn= BayesNode("myNode", parents, cpt)
    assert bn.p(True, event) == 0.95
    event = {'Burglary': False, 'Earthquake': True}
    assert bn.p(False, event)  == 0.71

    s = {'A': True, 'B': False, 'C': True, 'D': False}
    assert consistent_with(s, {})
    assert consistent_with(s, s)
    assert not consistent_with(s, {'A': False})
    assert not consistent_with(s, {'D': True})

    seed(21)
    p = rejection_sampling('Earthquake', {}, burglary, 1000)
    assert p[True], p[False] == (0.001, 0.999)

    seed(71); p = likelihood_weighting('Earthquake', {}, burglary, 1000)
    assert p[True], p[False] == (0.002, 0.998)

tests()
