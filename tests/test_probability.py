import pytest
from probability import *  # noqa


def tests():
    cpt = burglary.variable_node('Alarm')
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


def test_probdist_basic():
    P = ProbDist('Flip')
    P['H'], P['T'] = 0.25, 0.75
    assert P['H'] == 0.25


def test_probdist_frequency():
    P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    assert (P['lo'], P['med'], P['hi']) == (0.125, 0.375, 0.5)


def test_probdist_normalize():
    P = ProbDist('Flip')
    P['H'], P['T'] = 35, 65
    P = P.normalize()
    assert (P.prob['H'], P.prob['T']) == (0.350, 0.650)


def test_jointprob():
    P = JointProbDist(['X', 'Y'])
    P[1, 1] = 0.25
    assert P[1, 1] == 0.25
    P[dict(X=0, Y=1)] = 0.5
    assert P[dict(X=0, Y=1)] == 0.5


def test_event_values():
    assert event_values({'A': 10, 'B': 9, 'C': 8}, ['C', 'A']) == (8, 10)
    assert event_values((1, 2), ['C', 'A']) == (1, 2)


def test_enumerate_joint_ask():
    P = JointProbDist(['X', 'Y'])
    P[0, 0] = 0.25
    P[0, 1] = 0.5
    P[1, 1] = P[2, 1] = 0.125
    assert enumerate_joint_ask(
            'X', dict(Y=1), P).show_approx() == '0: 0.667, 1: 0.167, 2: 0.167'


def test_bayesnode_p():
    bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
    assert bn.p(False, {'Burglary': False, 'Earthquake': True}) == 0.375


def test_enumeration_ask():
    assert enumeration_ask(
            'Burglary', dict(JohnCalls=T, MaryCalls=T),
            burglary).show_approx() == 'False: 0.716, True: 0.284'


def test_elemination_ask():
    elimination_ask(
            'Burglary', dict(JohnCalls=T, MaryCalls=T),
            burglary).show_approx() == 'False: 0.716, True: 0.284'


def test_rejection_sampling():
    random.seed(47)
    rejection_sampling(
            'Burglary', dict(JohnCalls=T, MaryCalls=T),
            burglary, 10000).show_approx() == 'False: 0.7, True: 0.3'


def test_likelihood_weighting():
    random.seed(1017)
    assert likelihood_weighting(
            'Burglary', dict(JohnCalls=T, MaryCalls=T),
            burglary, 10000).show_approx() == 'False: 0.702, True: 0.298'

if __name__ == '__main__':
    pytest.main()
