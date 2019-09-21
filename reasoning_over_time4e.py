# Chapter 14 Probabilistic Reasoning Over Time
from utils4e import (
    element_wise_product, matrix_multiplication,
    vector_to_diagonal, vector_add, scalar_vector_product, inverse_matrix,
    weighted_sample_with_replacement, probability, normalize
)
import numpy as np

# ________________________________________________________________
# 14.2 Inference in Temporal Models
# 14.2.1 Filtering and prediction


def forward(HMM, fv, ev):
    prediction = vector_add(scalar_vector_product(fv[0], HMM.transition_model[0]),
                            scalar_vector_product(fv[1], HMM.transition_model[1]))
    sensor_dist = HMM.sensor_dist(ev)

    return normalize(element_wise_product(sensor_dist, prediction))

# 14.2.2 Smoothing


def backward(HMM, b, ev):
    sensor_dist = HMM.sensor_dist(ev)
    prediction = element_wise_product(sensor_dist, b)

    return normalize(vector_add(scalar_vector_product(prediction[0], HMM.transition_model[0]),
                                scalar_vector_product(prediction[1], HMM.transition_model[1])))


def forward_backward(HMM, ev, prior):
    """
    [Figure 14.4]. Forward-Backward algorithm for smoothing. Computes posterior probabilities
    of a sequence of states given a sequence of observations.
    :param HMM: a HiddenMarkovModel object
    :param ev: observed evidence, a list
    :param prior: prior probability, a list
    """
    t = len(ev)
    ev.insert(0, None)  # to make the code look similar to pseudo code

    fv = [[0.0, 0.0] for _ in range(len(ev))]
    b = [1.0, 1.0]
    bv = [b]    # we don't need bv; but we will have a list of all backward messages here
    sv = [[0, 0] for _ in range(len(ev))]

    fv[0] = prior

    for i in range(1, t + 1):
        fv[i] = forward(HMM, fv[i - 1], ev[i])
    for i in range(t, -1, -1):
        sv[i - 1] = normalize(element_wise_product(fv[i], b))
        b = backward(HMM, b, ev[i])
        bv.append(b)

    sv = sv[::-1]

    return sv


# 14.2.3 Finding the most likely sequence


def viterbi_algorithm(HMM, ev, prior):
    """
    Viterbi algorithm to find most likely hidden variable sequence
    :return the most likely hidden variable sequence, each variable represented by its most likely index
    """
    # time steps
    t = len(ev)
    # forward passing message and the most likely sequence initialization
    fv = [forward(HMM, prior, ev[0])]
    sequence = [fv[0].index(max(fv[0]))]

    # choose the most likely state at each time step
    for i in range(1, t):
        # extract sensor model
        sensor_dist = HMM.sensor_dist(ev[i])
        pre_seq = sequence[-1]
        # extract posterior probability on current evidence
        posterior = scalar_vector_product(fv[-1][pre_seq], HMM.transition_model[pre_seq])
        # get the most likely state
        p = element_wise_product(sensor_dist, posterior)
        fv.append(p)
        sequence.append(p.index(max(p)))
    return sequence


# ___________________________________________________________
# 14.3 Hidden Markov Models


class HiddenMarkovModel:
    """A Hidden markov model which takes Transition model and Sensor model as inputs"""

    def __init__(self, transition_model, sensor_model, prior=None):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.prior = prior or [0.5, 0.5]

    def sensor_dist(self, ev):
        if ev is True:
            return self.sensor_model[0]
        else:
            return self.sensor_model[1]

# _________________________________________________________________________
# 14.3.1 Simpliﬁed matrix algorithms


def fixed_lag_smoothing(e_t, HMM, d, ev, t):
    """
    [Figure 14.6]
    Smoothing algorithm with a fixed time lag of 'd' steps.
    Online algorithm that outputs the new smoothed estimate if observation
    for new time step is given.
    """
    ev.insert(0, None)

    T_model = HMM.transition_model
    f = HMM.prior
    B = [[1, 0], [0, 1]]
    evidence = []

    evidence.append(e_t)
    O_t = vector_to_diagonal(HMM.sensor_dist(e_t))
    if t > d:
        f = forward(HMM, f, e_t)
        O_tmd = vector_to_diagonal(HMM.sensor_dist(ev[t - d]))
        B = matrix_multiplication(inverse_matrix(O_tmd), inverse_matrix(T_model), B, T_model, O_t)
    else:
        B = matrix_multiplication(B, T_model, O_t)
    t += 1

    if t > d:
        # always returns a 1x2 matrix
        return [normalize(i) for i in matrix_multiplication([f], B)][0]
    else:
        return None

# _________________________________________________________________________
# 14.3.2 Hidden Markov model example: Localization
# localization example


class LocalizationExample:

    def __init__(self, available, error_rate=0.03):
        """
        The localization example in [Figure 14.7]
        :param available: a set of coordinates: {[0,0], [0,1]...}
        :param error_rate: error rate of each observation
        """
        self.available = available
        self.error_rate = error_rate
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def find_neighbors(self, x):
        """
        find the available neighbors of current point at x
        :param x: current point coordinate
        :return: a set of available neighbors
        """
        neighbors = set()
        if x in self.available:
            for d in self.directions:
                new_x = vector_add(d, x)
                if new_x in self.available:
                    neighbors.add(new_x)
        return neighbors

    def transition(self, cur, next):
        """
        transition model from state:current to state:next
        """
        neighbors = self.find_neighbors(cur)
        if next in neighbors:
            return 1/len(neighbors)
        else:
            return 0

    def observation(self, evidence, current):
        """
        The probability of the evidence if observed from current location
        :param evidence: directions in order of [n,s,w,e], if observed an obstacle, put 1 on that location
        :param current: coordinate of current location
        """
        dit = 0
        for i in range(len(self.directions)):
            if bool(1-evidence[i]) != bool(vector_add(current, self.directions[i]) in self.available):
                dit += 1  # different digit
        return (1-self.error_rate) ** (4-dit) * (self.error_rate ** dit)

    def locate(self, evidence):
        """
        calculate the probability of all available states
        :param: evidence: a list, a list of evidences in order of time step
        """
        possible_pos = dict()
        # init the probability of each state
        for x in self.available:
            possible_pos[x] = 1
        # calculate the probability of each position for each time step
        for e in evidence:
            nxt_possible_moves = {}
            for pos in possible_pos:
                possible_pos[pos] *= self.observation(e, pos)
                for n in self.find_neighbors(pos):
                    if self.transition(pos, n) * possible_pos[pos] != 0:
                        nxt_possible_moves[n] = max(self.transition(pos, n) * possible_pos[pos], nxt_possible_moves.get(n,0))
            res = possible_pos
            possible_pos = nxt_possible_moves
        return res


# _________________________________________________________________________
# 14.5 Dynamic Bayesian Networks

# battery examples

# accurate meter example
prior_a = [0, 0, 0, 0, 0, 1]
transient_model_a = [[0.95, 0.01, 0.01, 0.01, 0.01, 0.01], [0.26, 0.7, 0.01, 0.01, 0.01, 0.01],[0.01, 0.26, 0.7, 0.01, 0.01, 0.01],
                 [0.01, 0.01, 0.26, 0.7, 0.01,  0.01], [0.01, 0.01, 0.01, 0.26, 0.7, 0.01], [0.01, 0.01, 0.01, 0.01, 0.26, 0.7]]
sensor_model_a = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]

# transient failure model
prior_t = [0, 0, 0, 0, 0, 1]
transient_model_t = transient_model_a
sensor_model_t = [[1, 0.03, 0.03, 0.03, 0.03, 0.03], [0.03, 1, 0.03, 0.03, 0.03, 0.03],[0.03, 0.03, 1, 0.03, 0.03, 0.03],
                  [0.03, 0.03, 0.03, 1, 0.03, 0.03], [0.03, 0.03, 0.03, 0.03, 1, 0.03], [0.03, 0.03, 0.03, 0.03, 0.03, 1]]

# BMeter broken model in persistent failure model, hidden variable: BMBroken: [True, False]
prior_broken = [0, 1]
transient_model_broken = [[1, 0], [0.001, 0.999]]
sensor_model_broken = [[0.999, 0.001], [0.001, 0.999]]


#
def guess_battery(evidence, transit_model, sensor_model, prior):
    time_step = len(evidence)
    transit_model, sensor_model, prior = np.asarray(transit_model), np.asarray(sensor_model), np.asarray(prior)
    # transition probabilities of each time step
    trans = [prior]
    # expectation of each evidence
    expectation = []

    # do prediction
    for t in range(time_step):
        pre = trans[-1]
        ev = evidence[t]
        cur_trans = [pre[i] * transit_model[i] for i in range(len(prior))]
        cur_trans = normalize(np.sum(cur_trans, axis=0))

        # sensor model sliced by evidence
        cur_sens = sensor_model[ev]
        post_p = normalize(np.multiply(cur_sens, cur_trans))
        expectation.append(sum(k*i for k,i in enumerate(post_p)))
        trans.append(post_p)
    return expectation


# _________________________________________________________________________
# 14.5.3 Approximate inference in DBNs


def particle_filtering(e, N, HMM):
    """Particle filtering considering two states variables."""
    dist = [0.5, 0.5]
    # Weight Initialization
    w = [0 for _ in range(N)]
    # STEP 1
    # Propagate one step using transition model given prior state
    dist = vector_add(scalar_vector_product(dist[0], HMM.transition_model[0]),
                      scalar_vector_product(dist[1], HMM.transition_model[1]))
    # Assign state according to probability
    s = ['A' if probability(dist[0]) else 'B' for _ in range(N)]
    w_tot = 0
    # Calculate importance weight given evidence e
    for i in range(N):
        if s[i] == 'A':
            # P(U|A)*P(A)
            w_i = HMM.sensor_dist(e)[0] * dist[0]
        if s[i] == 'B':
            # P(U|B)*P(B)
            w_i = HMM.sensor_dist(e)[1] * dist[1]
        w[i] = w_i
        w_tot += w_i

    # Normalize all the weights
    for i in range(N):
        w[i] = w[i] / w_tot

    # Limit weights to 4 digits
    for i in range(N):
        w[i] = float("{0:.4f}".format(w[i]))

    # STEP 2

    s = weighted_sample_with_replacement(N, s, w)

    return s
