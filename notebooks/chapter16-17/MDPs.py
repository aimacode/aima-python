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

# %% [markdown]
# # MDP
#
# A sequential decision problem for a fully observable, stochastic environment with a Markovian transition model and additive rewards is called a Markov decision process. Markov model is very practical and can model many real-world decision-making processes. 
#
# Before we start playing with the actual implementations let us review a couple of key points about MDPs.
#
# - A stochastic process has the **Markov property** if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it.
#
#     -- Source: [Wikipedia](https://en.wikipedia.org/wiki/Markov_property)
#
# Often it is possible to model many different phenomena as a Markov process by being flexible with our definition of a state.
#    
#
# - MDPs help us deal with fully-observable and non-deterministic/stochastic environments. For dealing with partially-observable and stochastic cases we make use of a generalization of MDPs named POMDPs (partially observable Markov decision process).
#
# Our overall goal to solve an MDP is to come up with a policy that guides us to select the best action in each state so as to maximize the expected sum of future rewards.

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.mdp import *
from aima.notebook_utils import psource, pseudocode, plot_pomdp_utility

# %% [markdown]
# ## Implementation
#
# To begin with let us look at the implementation of the MDP class defined in mdp.py The docstring tells us what all is required to define an MDP namely - set of states, actions, initial state, transition model, and a reward function. Each of these is implemented as a method. Do not close the popup so that you can follow along with the description of the code below.

# %%
psource(MDP)

# %% [markdown]
# The **_ _init_ _** method takes in the following parameters:
#
# - init: the initial state.
# - actlist: List of actions possible in each state.
# - terminals: List of terminal states where only possible action is exit
# - gamma: Discounting factor. This makes sure that delayed rewards have less value compared to immediate ones.
#
# **R** method returns the reward for each state by using the self.reward dict.
#
# **T** method is not implemented and is somewhat different from the text. Here we return (probability, s') pairs which belong to list of possible state by taking action an in state s.
#
# **actions** method returns a list of actions possible in each state. By default, it returns all actions for states other than terminal states.
#

# %% [markdown]
# ## Example
#
# Now let us implement the simple MDP in the image below. States A, B have actions X, Y available in them. Their probabilities are shown just above the arrows. We start by using MDP as the base class for our CustomMDP. Obviously, we need to make a few changes to suit our case. We make use of a transition matrix as our transitions are not very simple.
# <img src="images/mdp-a.png">

# %%
# Transition Matrix as nested dict. State -> Actions in state -> List of (Probability, State) tuples
t = {
    "A": {
            "X": [(0.3, "A"), (0.7, "B")],
            "Y": [(1.0, "A")]
         },
    "B": {
            "X": {(0.8, "End"), (0.2, "B")},
            "Y": {(1.0, "A")}
         },
    "End": {}
}

init = "A"

terminals = ["End"]

rewards = {
    "A": 5,
    "B": -10,
    "End": 100
}


# %%
class CustomMDP(MDP):
    def __init__(self, init, terminals, transition_matrix, reward = None, gamma=.9):
        # All possible actions.
        actlist = []
        for state in transition_matrix.keys():
            actlist.extend(transition_matrix[state])
        actlist = list(set(actlist))
        MDP.__init__(self, init, actlist, terminals, transition_matrix, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else: 
            return self.t[state][action]


# %% [markdown]
# Finally we instantize the class with the parameters for our MDP in the picture.

# %%
our_mdp = CustomMDP(init, terminals, t, rewards, gamma=.9)

# %% [markdown]
# With this we have successfully represented our MDP. Later we will look at ways to solve this MDP.

# %% [markdown]
# # GRID MDP
#
# Now we look at a concrete implementation that makes use of the MDP as a base class. The GridMDP class in the MDP module is used to represent a grid world MDP like the one shown in **Fig 16.1** of the AIMA Book. We assume for now that the environment is _fully observable_ so that the agent always knows where it is. The code should be easy to understand if you have gone through the CustomMDP example.

# %%
psource(GridMDP)

# %% [markdown]
# The **_ _init_ _** method takes **grid** as an extra parameter compared to the MDP class. The grid is a nested list of rewards in states.
#
# **go** method returns the state by going in a particular direction by using vector_add.
#
# **T** method is not implemented and is somewhat different from the text. Here we return (probability, s') pairs belong to list of possible state by taking action an in state s.
#
# **actions** method returns a list of actions possible in each state. By default, it returns all actions for states other than terminal states.
#
# **to_arrows** are used for representing the policy in a grid-like format.

# %% [markdown]
# We can create a GridMDP like the one in **Fig 17.1** as follows: 
#
#     GridMDP([[-0.04, -0.04, -0.04, +1],
#             [-0.04, None,  -0.04, -1],
#             [-0.04, -0.04, -0.04, -0.04]],
#             terminals=[(3, 2), (3, 1)])
#             
# In fact, the **sequential_decision_environment** in the MDP module has been instantiated using the exact same code.

# %%
sequential_decision_environment

# %%
