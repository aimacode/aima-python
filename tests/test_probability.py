import pytest
from probability import *


def tests():
    cpt = burglary.variable_node('Alarm')
    parents = ['Burglary', 'Earthquake']
    event = {'Burglary': True, 'Earthquake': True}
    assert cpt.p(True, event) == 0.95
    event = {'Burglary': False, 'Earthquake': True}
    assert cpt.p(False, event) == 0.71
    # assert BoolCPT({T: 0.2, F: 0.625}).p(False, ['Burglary'], event) == 0.375
    # assert BoolCPT(0.75).p(False, [], {}) == 0.25
    # cpt = BoolCPT({True: 0.2, False: 0.7})
    # assert cpt.rand(['A'], {'A': True}) in [True, False]
    # cpt = BoolCPT({(True, True): 0.1, (True, False): 0.3,
    #                (False, True): 0.5, (False, False): 0.7})
    # assert cpt.rand(['A', 'B'], {'A': True, 'B': False}) in [True, False]
    # #enumeration_ask('Earthquake', {}, burglary)

    s = {'A': True, 'B': False, 'C': True, 'D': False}
    assert consistent_with(s, {})
    assert consistent_with(s, s)
    assert not consistent_with(s, {'A': False})
    assert not consistent_with(s, {'D': True})

    random.seed(21)
    p = rejection_sampling('Earthquake', {}, burglary, 1000)
    assert p[True], p[False] == (0.001, 0.999)

    random.seed(71)
    p = likelihood_weighting('Earthquake', {}, burglary, 1000)
    assert p[True], p[False] == (0.002, 0.998)

if __name__ == '__main__':
    pytest.main()
