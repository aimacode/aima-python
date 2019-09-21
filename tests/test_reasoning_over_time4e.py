from reasoning_over_time4e import *
from utils import rounder

T, F = True, False


def test_forward_backward():
    umbrella_prior = [0.5, 0.5]
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    umbrella_evidence = [T, T, F, T, T]
    assert (rounder(forward_backward(umbrellaHMM, umbrella_evidence, umbrella_prior)) ==
            [[0.6469, 0.3531], [0.8673, 0.1327], [0.8204, 0.1796], [0.3075, 0.6925],
             [0.8204, 0.1796], [0.8673, 0.1327]])

    umbrella_evidence = [T, F, T, F, T]
    assert rounder(forward_backward(umbrellaHMM, umbrella_evidence, umbrella_prior)) == [
            [0.5871, 0.4129], [0.7177, 0.2823], [0.2324, 0.7676], [0.6072, 0.3928],
            [0.2324, 0.7676], [0.7177, 0.2823]]


def test_viterbi_algorithm():
    umbrella_prior = [0.5, 0.5]
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    umbrella_evidence = [T, T, F, T, T]
    assert viterbi_algorithm(umbrellaHMM, umbrella_evidence, umbrella_prior) == [0, 0, 1, 0, 0]

    umbrella_evidence = [T, F, T, F, T]
    assert viterbi_algorithm(umbrellaHMM, umbrella_evidence, umbrella_prior) == [0, 1, 0, 1, 0]


def test_fixed_lag_smoothing():
    umbrella_evidence = [T, F, T, F, T]
    e_t = F
    t = 4
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.2], [0.1, 0.8]]
    umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

    d = 2
    assert rounder(fixed_lag_smoothing(e_t, umbrellaHMM, d,
                                       umbrella_evidence, t)) == [0.1111, 0.8889]
    d = 5
    assert fixed_lag_smoothing(e_t, umbrellaHMM, d, umbrella_evidence, t) is None

    umbrella_evidence = [T, T, F, T, T]
    # t = 4
    e_t = T

    d = 1
    assert rounder(fixed_lag_smoothing(e_t, umbrellaHMM,
                                       d, umbrella_evidence, t)) == [0.9939, 0.0061]

# test localization


available = {(0,0), (0, 1), (0,2), (0,3), (1,2), (1,3), (2,1), (2,2), (2,3), (3,0), (3,1), (3,3)}
localization_example = LocalizationExample(available)


def test_find_neighbors():
    assert localization_example.find_neighbors((0,0)) == {(0,1)}
    assert localization_example.find_neighbors((0,2)) == {(1, 2), (0, 3), (0, 1)}
    assert localization_example.find_neighbors((0,1)) == {(0, 0), (0, 2)}


def test_transition():
    assert localization_example.transition((0,0),(0,1)) == 1
    assert localization_example.transition((0,0), (0,2)) == 0
    assert localization_example.transition((1,0), (1,1,)) == 0
    assert localization_example.transition((1,2), (1,3)) == 1/3


def test_observation():
    assert localization_example.observation([1,1,1,0], (0,0)) == 0.8852928099999999
    assert localization_example.observation([0,1,1,1], (0,0)) == 0.0008468099999999999
    assert localization_example.observation([1,1,1,1], (3,0)) == 0.02738019


def test_locate():
    assert localization_example.locate([[1,1,1,0]])[(0, 0)] == 0.8852928099999999
    assert localization_example.locate([[1,1,1,0], [1,1,0,0]])[(0, 1)] == 0.783743359437696


# test battery example

def test_battery_example():
    assert guess_battery([1, 1, 1, 0, 0, 0, 1, 1, 1], transient_model_broken, sensor_model_broken, prior_broken) \
                         == [0.999998997998, 0.999998996993992, 0.999998996992986, 0.4997493736850905, 0.0009970059830369062, 9.979990039929959e-07, 0.000995013958127615, 0.4984987542583667, 0.9989919769972591]
    assert guess_battery([5, 5, 5, 0, 0, 0, 5, 5, 5], transient_model_a, sensor_model_a, prior_a) == [5, 5, 5, 0, 0, 0, 5, 5, 5]
    assert guess_battery([5,5,5,0,5,5,5],transient_model_t, sensor_model_t,  prior_t) == \
           [4.983074753173484, 4.982192650747305, 4.982148402050585, 3.4187123637458328, 4.8579622921495345, 4.976542474849849, 4.981902096279003]


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
