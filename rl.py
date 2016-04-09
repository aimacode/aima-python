"""Reinforcement Learning (Chapter 21)
"""

import agents

import random


class PassiveADPAgent(agents.Agent):

    """Passive (non-learning) agent that uses adaptive dynamic programming
    on a given MDP and policy. [Fig. 21.2]"""
    NotImplemented


class PassiveTDAgent:
    """The abstract class for a Passive (non-learning) agent that uses
    temporal differences to learn utility estimates. Override update_state
    method to convert percept to state and reward. The mdp being probided
    should be an instance of a subclass of the MDP Class.[Fig. 21.4]
    """

    def __init__(self, pi, mdp, alpha=None):

        self.pi = pi
        self.U = {s: 0. for s in mdp.states}
        self.Ns = {s: 0 for s in mdp.states}
        self.s = None
        self.a = None
        self.r = None
        self.gamma = mdp.gamma
        self.terminals = mdp.terminals

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1./(1+n)  # udacity video

    def __call__(self, percept):
        s_prime, r_prime = self.update_state(percept)
        pi, U, Ns, s, a, r = self.pi, self.U, self.Ns, self.s, self.a, self.r
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        if not Ns[s_prime]:
            U[s_prime] = r_prime
        if s is not None:
            Ns[s] += 1
            U[s] += alpha(Ns[s]) * (r + gamma * U[s_prime] - U[s])
        if s_prime in terminals:
            self.s = self.a = self.r = None
        else:
            self.s, self.a, self.r = s_prime, pi[s_prime], r_prime
        return self.a

    def update_state(self, percept):
        ''' To be overriden in most cases. The default case
        assumes th percept to be of type (state, reward)'''
        return percept


def run_single_trial(agent_program, mdp):
    ''' Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP '''

    def take_single_action(mdp, s, a):
        '''
        Selects outcome of taking action a
        in state s. Weighted Sampling.
        '''
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            cumulative_probability += probability
            if x < cumulative_probability:
                break
        return state

    current_state = mdp.init
    while True:
        current_reward = mdp.R(current_state)
        percept = (current_state, current_reward)
        next_action = agent_program(percept)
        if next_action is None:
            break
        current_state = take_single_action(mdp, current_state, next_action)
