# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run bootstrap.ipynb

# %% [markdown]
# # SARSA: on-policy temporal-difference control (Section 21.3)
#
# `SARSALearningAgent` from [`reinforcement_learning.py`](reinforcement_learning.py) is the on-policy counterpart of Q-learning: it bootstraps on the action its exploration policy actually takes, rather than the greedy maximum.

# %%
from aima.reinforcement_learning import SARSALearningAgent, QLearningAgent, run_single_trial
from aima.mdp import sequential_decision_environment as env

# %%
sarsa = SARSALearningAgent(env, Ne=5, Rplus=2, alpha=lambda n: 60. / (59 + n))
for _ in range(200):
    run_single_trial(sarsa, env)

print('Learned SARSA utilities U(s) = max_a Q(s, a):')
U = {}
for (state, action), q in sarsa.Q.items():
    if action is not None:
        U[state] = max(U.get(state, -float('inf')), q)
for state in sorted(U):
    print(' ', state, '->', round(U[state], 3))
