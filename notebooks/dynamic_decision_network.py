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
# # Dynamic Decision Networks (Section 17.4)
#
# Online decision making for a POMDP modeled as a dynamic decision network, using the belief-state look-ahead agent in [`mdp4e.py`](mdp4e.py).

# %%
from aima.mdp import POMDP, update_belief, pomdp_lookahead

# %% [markdown]
# A two-state 'tiger-like' POMDP: action 0 pays off in state 0, action 1 in state 1, and action 2 is a sensing action (small cost, informative observation).

# %%
t_prob = [[[0.65, 0.35], [0.65, 0.35]], [[0.65, 0.35], [0.65, 0.35]], [[1.0, 0.0], [0.0, 1.0]]]
e_prob = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.8, 0.2], [0.3, 0.7]]]
rewards = [[5, -10], [-20, 5], [-1, -1]]
pomdp = POMDP(('0', '1', '2'), t_prob, e_prob, rewards, ('0', '1'), gamma=0.95)

# %% [markdown]
# ### Belief update (POMDP filtering, Equation 17.17)
# Observation 0 (more likely in state 0) shifts a uniform belief towards state 0.

# %%
print('belief after sensing obs 0:', [round(b, 3) for b in update_belief(pomdp, [0.5, 0.5], '2', 0)])

# %% [markdown]
# ### Look-ahead decisions
# When the state is known the agent commits to the rewarding action; when it is unknown it prefers to gather information first.

# %%
print('belief [0.9, 0.1], depth 1 -> action', pomdp_lookahead(pomdp, [0.9, 0.1], depth=1))
print('belief [0.1, 0.9], depth 1 -> action', pomdp_lookahead(pomdp, [0.1, 0.9], depth=1))
print('belief [0.5, 0.5], depth 2 -> action', pomdp_lookahead(pomdp, [0.5, 0.5], depth=2), '(sense)')
