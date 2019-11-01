import pytest

from probability import *
from utils import rounder

random.seed("aima-python")


def tests():
    cpt = burglary.variable_node('Alarm')
    event = {'Burglary': True, 'Earthquake': True}
    assert cpt.p(True, event) == 0.95
    event = {'Burglary': False, 'Earthquake': True}
    assert cpt.p(False, event) == 0.71
    # enumeration_ask('Earthquake', {}, burglary)

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
    assert P['T'] == 0.75
    assert P['X'] == 0.00

    P = ProbDist('BiasedDie')
    P['1'], P['2'], P['3'], P['4'], P['5'], P['6'] = 10, 15, 25, 30, 40, 80
    P.normalize()
    assert P['2'] == 0.075
    assert P['4'] == 0.15
    assert P['6'] == 0.4


def test_probdist_frequency():
    P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    assert (P['lo'], P['med'], P['hi']) == (0.125, 0.375, 0.5)

    P = ProbDist('Pascal-5', {'x1': 1, 'x2': 5, 'x3': 10, 'x4': 10, 'x5': 5, 'x6': 1})
    assert (P['x1'], P['x2'], P['x3'], P['x4'], P['x5'], P['x6']) == (
        0.03125, 0.15625, 0.3125, 0.3125, 0.15625, 0.03125)


def test_probdist_normalize():
    P = ProbDist('Flip')
    P['H'], P['T'] = 35, 65
    P = P.normalize()
    assert (P.prob['H'], P.prob['T']) == (0.350, 0.650)

    P = ProbDist('BiasedDie')
    P['1'], P['2'], P['3'], P['4'], P['5'], P['6'] = 10, 15, 25, 30, 40, 80
    P = P.normalize()
    assert (P.prob['1'], P.prob['2'], P.prob['3'], P.prob['4'], P.prob['5'], P.prob['6']) == (
        0.05, 0.075, 0.125, 0.15, 0.2, 0.4)


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

    Q = JointProbDist(['W', 'X', 'Y', 'Z'])
    Q[0, 1, 1, 0] = 0.12
    Q[1, 0, 1, 1] = 0.4
    Q[0, 0, 1, 1] = 0.5
    Q[0, 0, 1, 0] = 0.05
    Q[0, 0, 0, 0] = 0.675
    Q[1, 1, 1, 0] = 0.3
    assert enumerate_joint(['W'], dict(X=0, Y=0, Z=1), Q) == 0
    assert enumerate_joint(['W'], dict(X=0, Y=0, Z=0), Q) == 0.675
    assert enumerate_joint(['W'], dict(X=0, Y=1, Z=1), Q) == 0.9
    assert enumerate_joint(['Y'], dict(W=1, X=0, Z=1), Q) == 0.4
    assert enumerate_joint(['Z'], dict(W=0, X=0, Y=0), Q) == 0.675
    assert enumerate_joint(['Z'], dict(W=1, X=1, Y=1), Q) == 0.3


def test_enumerate_joint_ask():
    P = JointProbDist(['X', 'Y'])
    P[0, 0] = 0.25
    P[0, 1] = 0.5
    P[1, 1] = P[2, 1] = 0.125
    assert enumerate_joint_ask(
        'X', dict(Y=1), P).show_approx() == '0: 0.667, 1: 0.167, 2: 0.167'


def test_bayesnode_p():
    bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
    assert bn.p(True, {'Burglary': True, 'Earthquake': False}) == 0.2
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
    assert enumeration_ask(
        'Burglary', dict(JohnCalls=T, MaryCalls=F),
        burglary).show_approx() == 'False: 0.995, True: 0.00513'
    assert enumeration_ask(
        'Burglary', dict(JohnCalls=F, MaryCalls=T),
        burglary).show_approx() == 'False: 0.993, True: 0.00688'
    assert enumeration_ask(
        'Burglary', dict(JohnCalls=T),
        burglary).show_approx() == 'False: 0.984, True: 0.0163'
    assert enumeration_ask(
        'Burglary', dict(MaryCalls=T),
        burglary).show_approx() == 'False: 0.944, True: 0.0561'


def test_elimination_ask():
    assert elimination_ask(
        'Burglary', dict(JohnCalls=T, MaryCalls=T),
        burglary).show_approx() == 'False: 0.716, True: 0.284'
    assert elimination_ask(
        'Burglary', dict(JohnCalls=T, MaryCalls=F),
        burglary).show_approx() == 'False: 0.995, True: 0.00513'
    assert elimination_ask(
        'Burglary', dict(JohnCalls=F, MaryCalls=T),
        burglary).show_approx() == 'False: 0.993, True: 0.00688'
    assert elimination_ask(
        'Burglary', dict(JohnCalls=T),
        burglary).show_approx() == 'False: 0.984, True: 0.0163'
    assert elimination_ask(
        'Burglary', dict(MaryCalls=T),
        burglary).show_approx() == 'False: 0.944, True: 0.0561'


def test_prior_sample():
    random.seed(42)
    all_obs = [prior_sample(burglary) for x in range(1000)]
    john_calls_true = [observation for observation in all_obs if observation['JohnCalls']]
    mary_calls_true = [observation for observation in all_obs if observation['MaryCalls']]
    burglary_and_john = [observation for observation in john_calls_true if observation['Burglary']]
    burglary_and_mary = [observation for observation in mary_calls_true if observation['Burglary']]
    assert len(john_calls_true) / 1000 == 46 / 1000
    assert len(mary_calls_true) / 1000 == 13 / 1000
    assert len(burglary_and_john) / len(john_calls_true) == 1 / 46
    assert len(burglary_and_mary) / len(mary_calls_true) == 1 / 13


def test_prior_sample2():
    random.seed(128)
    all_obs = [prior_sample(sprinkler) for x in range(1000)]
    rain_true = [observation for observation in all_obs if observation['Rain']]
    sprinkler_true = [observation for observation in all_obs if observation['Sprinkler']]
    rain_and_cloudy = [observation for observation in rain_true if observation['Cloudy']]
    sprinkler_and_cloudy = [observation for observation in sprinkler_true if observation['Cloudy']]
    assert len(rain_true) / 1000 == 0.476
    assert len(sprinkler_true) / 1000 == 0.291
    assert len(rain_and_cloudy) / len(rain_true) == 376 / 476
    assert len(sprinkler_and_cloudy) / len(sprinkler_true) == 39 / 291


def test_rejection_sampling():
    random.seed(47)
    assert rejection_sampling(
        'Burglary', dict(JohnCalls=T, MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.7, True: 0.3'
    assert rejection_sampling(
        'Burglary', dict(JohnCalls=T, MaryCalls=F),
        burglary, 10000).show_approx() == 'False: 1, True: 0'
    assert rejection_sampling(
        'Burglary', dict(JohnCalls=F, MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.987, True: 0.0128'
    assert rejection_sampling(
        'Burglary', dict(JohnCalls=T),
        burglary, 10000).show_approx() == 'False: 0.982, True: 0.0183'
    assert rejection_sampling(
        'Burglary', dict(MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.965, True: 0.0348'


def test_rejection_sampling2():
    random.seed(42)
    assert rejection_sampling(
        'Cloudy', dict(Rain=T, Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.56, True: 0.44'
    assert rejection_sampling(
        'Cloudy', dict(Rain=T, Sprinkler=F),
        sprinkler, 10000).show_approx() == 'False: 0.119, True: 0.881'
    assert rejection_sampling(
        'Cloudy', dict(Rain=F, Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.951, True: 0.049'
    assert rejection_sampling(
        'Cloudy', dict(Rain=T),
        sprinkler, 10000).show_approx() == 'False: 0.205, True: 0.795'
    assert rejection_sampling(
        'Cloudy', dict(Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.835, True: 0.165'


def test_likelihood_weighting():
    random.seed(1017)
    assert likelihood_weighting(
        'Burglary', dict(JohnCalls=T, MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.702, True: 0.298'
    assert likelihood_weighting(
        'Burglary', dict(JohnCalls=T, MaryCalls=F),
        burglary, 10000).show_approx() == 'False: 0.993, True: 0.00656'
    assert likelihood_weighting(
        'Burglary', dict(JohnCalls=F, MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.996, True: 0.00363'
    assert likelihood_weighting(
        'Burglary', dict(JohnCalls=F, MaryCalls=F),
        burglary, 10000).show_approx() == 'False: 1, True: 0.000126'
    assert likelihood_weighting(
        'Burglary', dict(JohnCalls=T),
        burglary, 10000).show_approx() == 'False: 0.979, True: 0.0205'
    assert likelihood_weighting(
        'Burglary', dict(MaryCalls=T),
        burglary, 10000).show_approx() == 'False: 0.94, True: 0.0601'


def test_likelihood_weighting2():
    random.seed(42)
    assert likelihood_weighting(
        'Cloudy', dict(Rain=T, Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.559, True: 0.441'
    assert likelihood_weighting(
        'Cloudy', dict(Rain=T, Sprinkler=F),
        sprinkler, 10000).show_approx() == 'False: 0.12, True: 0.88'
    assert likelihood_weighting(
        'Cloudy', dict(Rain=F, Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.951, True: 0.0486'
    assert likelihood_weighting(
        'Cloudy', dict(Rain=T),
        sprinkler, 10000).show_approx() == 'False: 0.198, True: 0.802'
    assert likelihood_weighting(
        'Cloudy', dict(Sprinkler=T),
        sprinkler, 10000).show_approx() == 'False: 0.833, True: 0.167'


def test_forward_backward():
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    umbrella_evidence = [T, T, F, T, T]
    assert rounder(forward_backward(umbrellaHMM, umbrella_evidence)) == [
        [0.6469, 0.3531], [0.8673, 0.1327], [0.8204, 0.1796], [0.3075, 0.6925], [0.8204, 0.1796], [0.8673, 0.1327]]

    umbrella_evidence = [T, F, T, F, T]
    assert rounder(forward_backward(umbrellaHMM, umbrella_evidence)) == [
        [0.5871, 0.4129], [0.7177, 0.2823], [0.2324, 0.7676], [0.6072, 0.3928], [0.2324, 0.7676], [0.7177, 0.2823]]


def test_viterbi():
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    umbrella_evidence = [T, T, F, T, T]
    assert viterbi(umbrellaHMM, umbrella_evidence)[0] == [T, T, F, T, T]
    assert rounder(viterbi(umbrellaHMM, umbrella_evidence)[1]) == [0.8182, 0.5155, 0.1237, 0.0334, 0.0210]

    umbrella_evidence = [T, F, T, F, T]
    assert viterbi(umbrellaHMM, umbrella_evidence)[0] == [T, F, F, F, T]
    assert rounder(viterbi(umbrellaHMM, umbrella_evidence)[1]) == [0.8182, 0.1964, 0.0275, 0.0154, 0.0042]


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
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)
    s = particle_filtering(umbrella_evidence, N, umbrellaHMM)
    assert len(s) == N
    assert all(state in 'AB' for state in s)
    # XXX 'A' and 'B' are really arbitrary names, but I'm letting it stand for now


def test_monte_carlo_localization():
    # TODO: Add tests for random motion/inaccurate sensors
    random.seed('aima-python')
    m = MCLmap([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]])

    def P_motion_sample(kin_state, v, w):
        """Sample from possible kinematic states.
        Returns from a single element distribution (no uncertainty in motion)"""
        pos = kin_state[:2]
        orient = kin_state[2]

        # for simplicity the robot first rotates and then moves
        orient = (orient + w) % 4
        for _ in range(orient):
            v = (v[1], -v[0])
        pos = vector_add(pos, v)
        return pos + (orient,)

    def P_sensor(x, y):
        """Conditional probability for sensor reading"""
        # Need not be exact probability. Can use a scaled value.
        if x == y:
            return 0.8
        elif abs(x - y) <= 2:
            return 0.05
        else:
            return 0

    from utils import print_table
    a = {'v': (0, 0), 'w': 0}
    z = (2, 4, 1, 6)
    S = monte_carlo_localization(a, z, 1000, P_motion_sample, P_sensor, m)
    grid = [[0] * 17 for _ in range(11)]
    for x, y, _ in S:
        if 0 <= x < 11 and 0 <= y < 17:
            grid[x][y] += 1
    print("GRID:")
    print_table(grid)

    a = {'v': (0, 1), 'w': 0}
    z = (2, 3, 5, 7)
    S = monte_carlo_localization(a, z, 1000, P_motion_sample, P_sensor, m, S)
    grid = [[0] * 17 for _ in range(11)]
    for x, y, _ in S:
        if 0 <= x < 11 and 0 <= y < 17:
            grid[x][y] += 1
    print("GRID:")
    print_table(grid)

    assert grid[6][7] > 700


def test_gibbs_ask():
    possible_solutions = ['False: 0.16, True: 0.84', 'False: 0.17, True: 0.83', 'False: 0.15, True: 0.85']
    g_solution = gibbs_ask('Cloudy', dict(Rain=True), sprinkler, 200).show_approx()
    assert g_solution in possible_solutions


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

# A Joint Probability Distribution is dealt with like this [Figure 13.3]:
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
