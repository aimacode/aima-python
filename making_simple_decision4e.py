"""Making Simple Decisions (Chapter 15)"""

import random

from agents import Agent
from probability import BayesNet
from utils4e import vector_add, weighted_sample_with_replacement


class DecisionNetwork(BayesNet):
    """An abstract class for a decision network as a wrapper for a BayesNet.
    Represents an agent's current state, its possible actions, reachable states
    and utilities of those states."""

    def __init__(self, action, infer):
        """action: a single action node
        infer: the preferred method to carry out inference on the given BayesNet"""
        super().__init__()
        self.action = action
        self.infer = infer

    def best_action(self):
        """Return the best action in the network"""
        return self.action

    def get_utility(self, action, state):
        """Return the utility for a particular action and state in the network"""
        raise NotImplementedError

    def get_expected_utility(self, action, evidence):
        """Compute the expected utility given an action and evidence"""
        u = 0.0
        prob_dist = self.infer(action, evidence, self).prob
        for item, _ in prob_dist.items():
            u += prob_dist[item] * self.get_utility(action, item)

        return u


class InformationGatheringAgent(Agent):
    """A simple information gathering agent. The agent works by repeatedly selecting
    the observation with the highest information value, until the cost of the next
    observation is greater than its expected benefit. [Figure 16.9]"""

    def __init__(self, decnet, infer, initial_evidence=None):
        """decnet: a decision network
        infer: the preferred method to carry out inference on the given decision network
        initial_evidence: initial evidence"""
        super().__init__()
        self.decnet = decnet
        self.infer = infer
        self.observation = initial_evidence or []
        self.variables = self.decnet.nodes

    def integrate_percept(self, percept):
        """Integrate the given percept into the decision network"""
        raise NotImplementedError

    def execute(self, percept):
        """Execute the information gathering algorithm"""
        self.observation = self.integrate_percept(percept)
        vpis = self.vpi_cost_ratio(self.variables)
        j = max(vpis)
        variable = self.variables[j]

        if self.vpi(variable) > self.cost(variable):
            return self.request(variable)

        return self.decnet.best_action()

    def request(self, variable):
        """Return the value of the given random variable as the next percept"""
        raise NotImplementedError

    def cost(self, var):
        """Return the cost of obtaining evidence through tests, consultants or questions"""
        raise NotImplementedError

    def vpi_cost_ratio(self, variables):
        """Return the VPI to cost ratio for the given variables"""
        v_by_c = []
        for var in variables:
            v_by_c.append(self.vpi(var) / self.cost(var))
        return v_by_c

    def vpi(self, variable):
        """Return VPI for a given variable"""
        vpi = 0.0
        prob_dist = self.infer(variable, self.observation, self.decnet).prob
        for item, _ in prob_dist.items():
            post_prob = prob_dist[item]
            new_observation = list(self.observation)
            new_observation.append(item)
            expected_utility = self.decnet.get_expected_utility(variable, new_observation)
            vpi += post_prob * expected_utility

        vpi -= self.decnet.get_expected_utility(variable, self.observation)
        return vpi


# _________________________________________________________________________
# chapter 25 Robotics
# TODO: Implement continuous map for MonteCarlo similar to Fig25.10 from the book


class MCLmap:
    """Map which provides probability distributions and sensor readings.
    Consists of discrete cells which are either an obstacle or empty"""

    def __init__(self, m):
        self.m = m
        self.nrows = len(m)
        self.ncols = len(m[0])
        # list of empty spaces in the map
        self.empty = [(i, j) for i in range(self.nrows) for j in range(self.ncols) if not m[i][j]]

    def sample(self):
        """Returns a random kinematic state possible in the map"""
        pos = random.choice(self.empty)
        # 0N 1E 2S 3W
        orient = random.choice(range(4))
        kin_state = pos + (orient,)
        return kin_state

    def ray_cast(self, sensor_num, kin_state):
        """Returns distace to nearest obstacle or map boundary in the direction of sensor"""
        pos = kin_state[:2]
        orient = kin_state[2]
        # sensor layout when orientation is 0 (towards North)
        #  0
        # 3R1
        #  2
        delta = ((sensor_num % 2 == 0) * (sensor_num - 1), (sensor_num % 2 == 1) * (2 - sensor_num))
        # sensor direction changes based on orientation
        for _ in range(orient):
            delta = (delta[1], -delta[0])
        range_count = 0
        while (0 <= pos[0] < self.nrows) and (0 <= pos[1] < self.nrows) and (not self.m[pos[0]][pos[1]]):
            pos = vector_add(pos, delta)
            range_count += 1
        return range_count


def monte_carlo_localization(a, z, N, P_motion_sample, P_sensor, m, S=None):
    """Monte Carlo localization algorithm from Fig 25.9"""

    def ray_cast(sensor_num, kin_state, m):
        return m.ray_cast(sensor_num, kin_state)

    M = len(z)
    W = [0] * N
    S_ = [0] * N
    W_ = [0] * N
    v = a['v']
    w = a['w']

    if S is None:
        S = [m.sample() for _ in range(N)]

    for i in range(N):
        S_[i] = P_motion_sample(S[i], v, w)
        W_[i] = 1
        for j in range(M):
            z_ = ray_cast(j, S_[i], m)
            W_[i] = W_[i] * P_sensor(z[j], z_)

    S = weighted_sample_with_replacement(N, S_, W_)
    return S
