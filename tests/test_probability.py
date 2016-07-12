import pytest
import random
from probability import *  # noqa
from utils import rounder


def tests():
    cpt = burglary.variable_node('Alarm')
    event = {'Burglary': True, 'Earthquake': True}
    assert cpt.p(True, event) == 0.95
    event = {'Burglary': False, 'Earthquake': True}
    assert cpt.p(False, event) == 0.71
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


def test_enumerate_joint():
    P = JointProbDist(['X', 'Y'])
    P[0, 0] = 0.25
    P[0, 1] = 0.5
    P[1, 1] = P[2, 1] = 0.125
    assert enumerate_joint(['Y'], dict(X=0), P) == 0.75
    assert enumerate_joint(['X'], dict(Y=2), P) == 0
    assert enumerate_joint(['X'], dict(Y=1), P) == 0.75


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
    assert BayesNode('W', '', 0.75).p(False, {'Random': True}) == 0.25


def test_bayesnode_sample():
    X = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
    assert X.sample({'Burglary': False, 'Earthquake': True}) in [True, False]
    Z = BayesNode('Z', 'P Q', {(True, True): 0.2, (True, False): 0.3,
                               (False, True): 0.5, (False, False): 0.7})
    assert Z.sample({'P': True, 'Q': False}) in [True, False]


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


def test_forward_backward():
    umbrella_prior = [0.5, 0.5]
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    umbrella_evidence = [T, T, F, T, T]
    assert (rounder(forward_backward(umbrellaHMM, umbrella_evidence, umbrella_prior)) ==
            [[0.6469, 0.3531], [0.8673, 0.1327], [0.8204, 0.1796], [0.3075, 0.6925], [0.8204, 0.1796], [0.8673, 0.1327]])

    umbrella_evidence = [T, F, T, F, T]
    assert rounder(forward_backward(umbrellaHMM, umbrella_evidence, umbrella_prior)) == [[0.5871, 0.4129],
                 [0.7177, 0.2823], [0.2324, 0.7676], [0.6072, 0.3928], [0.2324, 0.7676], [0.7177, 0.2823]]


def test_fixed_lag_smoothing():
    umbrella_evidence = [T, F, T, F, T]
    e_t = F
    t = 4
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    d = 2
    assert rounder(fixed_lag_smoothing(e_t, umbrellaHMM, d, umbrella_evidence, t)) == [0.1111, 0.8889]
    d = 5
    assert fixed_lag_smoothing(e_t, umbrellaHMM, d, umbrella_evidence, t) is None

    umbrella_evidence = [T, T, F, T, T]
    # t = 4
    e_t = T

    d = 1
    assert rounder(fixed_lag_smoothing(e_t, umbrellaHMM, d, umbrella_evidence, t)) == [0.9939, 0.0061]


def test_particle_filtering():
    N = 10
    umbrella_evidence = T
    umbrella_prior = [0.5, 0.5]
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)
    s = particle_filtering(umbrella_evidence, N, umbrellaHMM)
    assert len(s) == N
    assert all(state in 'AB' for state in s)
    # XXX 'A' and 'B' are really arbitrary names, but I'm letting it stand for now


# The following should probably go in .ipynb:

"""
# We can build up a probability distribution like this (p. 469):
>>> P = ProbDist()
>>> P['sunny'] = 0.7
>>> P['rain'] = 0.2
>>> P['cloudy'] = 0.08
>>> P['snow'] = 0.02

# and query it like this:  (Never mind this ELLIPSIS option
#                           added to make the doctest portable.)
>>> P['rain']               #doctest:+ELLIPSIS
0.2...

# A Joint Probability Distribution is dealt with like this [Figure 13.3]:  # noqa
>>> P = JointProbDist(['Toothache', 'Cavity', 'Catch'])
>>> T, F = True, False
>>> P[T, T, T] = 0.108; P[T, T, F] = 0.012; P[F, T, T] = 0.072; P[F, T, F] = 0.008
>>> P[T, F, T] = 0.016; P[T, F, F] = 0.064; P[F, F, T] = 0.144; P[F, F, F] = 0.576

>>> P[T, T, T]
0.108

# Ask for P(Cavity|Toothache=T)
>>> PC = enumerate_joint_ask('Cavity', {'Toothache': T}, P)
>>> PC.show_approx()
'False: 0.4, True: 0.6'

>>> 0.6-epsilon < PC[T] < 0.6+epsilon
True

>>> 0.4-epsilon < PC[F] < 0.4+epsilon
True
"""

if __name__ == '__main__':
    pytest.main()
