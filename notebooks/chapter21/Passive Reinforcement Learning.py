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
# # Introduction to Reinforcement Learning
#
# This Jupyter notebook and the others in the same folder act as supporting materials for **Chapter 21 Reinforcement Learning** of the book* Artificial Intelligence: A Modern Approach*. The notebooks make use of the implementations in `rl.py` module. We also make use of the implementation of MDPs in the `mdp.py` module to test our agents. It might be helpful if you have already gone through the Jupyter notebook dealing with the Markov decision process. Let us import everything from the `rl` module. It might be helpful to view the source of some of our implementations.

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.reinforcement_learning import *

# %% [markdown]
# Before we start playing with the actual implementations let us review a couple of things about RL.
#
# 1. Reinforcement Learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. 
#
# 2. Reinforcement learning differs from standard supervised learning in that correct input/output pairs are never presented, nor sub-optimal actions explicitly corrected. Further, there is a focus on on-line performance, which involves finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).
#
# -- Source: [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
#
# In summary, we have a sequence of state action transitions with rewards associated with some states. Our goal is to find the optimal policy $\pi$ which tells us what action to take in each state.

# %% [markdown]
# # Passive Reinforcement Learning
#
# In passive Reinforcement Learning the agent follows a fixed policy $\pi$. Passive learning attempts to evaluate the given policy $pi$ - without any knowledge of the Reward function $R(s)$ and the Transition model $P(s'\ |\ s, a)$.
#
# This is usually done by some method of **utility estimation**. The agent attempts to directly learn the utility of each state that would result from following the policy. Note that at each step, it has to *perceive* the reward and the state - it has no global knowledge of these. Thus, if a certain the entire set of actions offers a very low probability of attaining some state $s_+$ - the agent may never perceive the reward $R(s_+)$.
#
# Consider a situation where an agent is given the policy to follow. Thus, at any point, it knows only its current state and current reward, and the action it must take next. This action may lead it to more than one state, with different probabilities.
#
# For a series of actions given by $\pi$, the estimated utility $U$:
# $$U^{\pi}(s) = E(\sum_{t=0}^\inf \gamma^t R^t(s'))$$
# Or the expected value of summed discounted rewards until termination.
#
# Based on this concept, we discuss three methods of estimating utility: direct utility estimation, adaptive dynamic programming, and temporal-difference learning.
#
# ### Implementation
#
# Passive agents are implemented in `rl4e.py` as various `Agent-Class`es.
#
# To demonstrate these agents, we make use of the `GridMDP` object from the `MDP` module. `sequential_decision_environment` is similar to that used for the `MDP` notebook but has discounting with $\gamma = 0.9$.
#
# The `Agent-Program` can be obtained by creating an instance of the relevant `Agent-Class`. The `__call__` method allows the `Agent-Class` to be called as a function. The class needs to be instantiated with a policy ($\pi$) and an `MDP` whose utility of states will be estimated.
#

# %%
from aima.mdp import sequential_decision_environment

# %% [markdown]
# The `sequential_decision_environment` is a GridMDP object as shown below. The rewards are **+1** and **-1** in the terminal states, and **-0.04** in the rest. <img src="images/mdp.png"> Now we define actions and a policy similar to **Fig 21.1** in the book.

# %%
# Action Directions
north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

policy = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north,                (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}

# %% [markdown]
# This enviroment will be extensively used in the following demonstrations.

# %% [markdown]
# ## Direct Utility Estimation (DUE)
#  
#  The first, most naive method of estimating utility comes from the simplest interpretation of the above definition. We construct an agent that follows the policy until it reaches the terminal state. At each step, it logs its current state, reward. Once it reaches the terminal state, it can estimate the utility for each state for *that* iteration, by simply summing the discounted rewards from that state to the terminal one.
#
#  It can now run this 'simulation' $n$ times and calculate the average utility of each state. If a state occurs more than once in a simulation, both its utility values are counted separately.
#  
#  Note that this method may be prohibitively slow for very large state-spaces. Besides, **it pays no attention to the transition probability $P(s'\ |\ s, a)$.** It misses out on information that it is capable of collecting (say, by recording the number of times an action from one state led to another state). The next method addresses this issue.
#  
# ### Examples
#
# The `PassiveDEUAgent` class in the `rl` module implements the Agent Program described in **Fig 21.2** of the AIMA Book. `PassiveDEUAgent` sums over rewards to find the estimated utility for each state. It thus requires the running of several iterations.

# %%
# %psource PassiveDUEAgent

# %% [markdown]
# Now let's try the `PassiveDEUAgent` on the newly defined `sequential_decision_environment`:

# %%
DUEagent = PassiveDUEAgent(policy, sequential_decision_environment)

# %% [markdown]
# We can try passing information through the markove model for 200 times in order to get the converged utility value:

# %%
for i in range(200):
    run_single_trial(DUEagent, sequential_decision_environment)
    DUEagent.estimate_U()

# %% [markdown]
# Now let's print our estimated utility for each position:

# %%
print('\n'.join([str(k)+':'+str(v) for k, v in DUEagent.U.items()]))

# %% [markdown]
# ## Adaptive Dynamic Programming (ADP)
#  
#  This method makes use of knowledge of the past state $s$, the action $a$, and the new perceived state $s'$ to estimate the transition probability $P(s'\ |\ s,a)$. It does this by the simple counting of new states resulting from previous states and actions.<br> 
#  The program runs through the policy a number of times, keeping track of:
#     - each occurrence of state $s$ and the policy-recommended action $a$ in $N_{sa}$
#     - each occurrence of $s'$ resulting from $a$ on $s$ in $N_{s'|sa}$.
#      
#  It can thus estimate $P(s'\ |\ s,a)$ as $N_{s'|sa}/N_{sa}$, which in the limit of infinite trials, will converge to the true value.<br>
#  Using the transition probabilities thus estimated, it can apply `POLICY-EVALUATION` to estimate the utilities $U(s)$ using properties of convergence of the Bellman functions.
#  
# ### Examples
#
# The `PassiveADPAgent` class in the `rl` module implements the Agent Program described in **Fig 21.2** of the AIMA Book. `PassiveADPAgent` uses state transition and occurrence counts to estimate $P$, and then $U$. Go through the source below to understand the agent.

# %%
# %psource

# %% [markdown]
# We instantiate a `PassiveADPAgent` below with the `GridMDP` shown and train it for 200 steps. The `rl` module has a simple implementation to simulate a single step of the iteration. The function is called `run_single_trial`.

# %%
ADPagent = PassiveADPAgent(policy, sequential_decision_environment)
for i in range(200):
    run_single_trial(ADPagent, sequential_decision_environment)

# %% [markdown]
# The utilities are calculated as :

# %%
print('\n'.join([str(k)+':'+str(v) for k, v in ADPagent.U.items()]))

# %% [markdown]
# When comparing to the result of `PassiveDUEAgent`, they both have -1.0 for utility at (3,1) and 1.0 at (3,2). Another point to notice is that the spot with the highest utility for both agents is (2,2) beside the terminal states, which is easy to understand when referring to the map.

# %% [markdown]
# ## Temporal-difference learning (TD)
#  
#  Instead of explicitly building the transition model $P$, the temporal-difference model makes use of the expected closeness between the utilities of two consecutive states $s$ and $s'$.
#  For the transition $s$ to $s'$, the update is written as:
# $$U^{\pi}(s) \leftarrow U^{\pi}(s) + \alpha \left( R(s) + \gamma U^{\pi}(s') - U^{\pi}(s) \right)$$
#  This model implicitly incorporates the transition probabilities by being weighed for each state by the number of times it is achieved from the current state. Thus, over a number of iterations, it converges similarly to the Bellman equations.
#  The advantage of the TD learning model is its relatively simple computation at each step, rather than having to keep track of various counts.
#  For $n_s$ states and $n_a$ actions the ADP model would have $n_s \times n_a$ numbers $N_{sa}$ and $n_s^2 \times n_a$ numbers $N_{s'|sa}$ to keep track of. The TD model must only keep track of a utility $U(s)$ for each state.
#  
# ### Examples
#
# `PassiveTDAgent` uses temporal differences to learn utility estimates. We learn the difference between the states and back up the values to previous states.  Let us look into the source before we see some usage examples.

# %%
# %psource PassiveTDAgent

# %% [markdown]
# In creating the `TDAgent`, we use the **same learning rate** $\alpha$ as given in the footnote of the book: $\alpha(n)=60/(59+n)$

# %%
TDagent = PassiveTDAgent(policy, sequential_decision_environment, alpha = lambda n: 60./(59+n))

# %% [markdown]
# Now we run **200 trials** for the agent to estimate Utilities.

# %%
for i in range(200):
    run_single_trial(TDagent,sequential_decision_environment)

# %% [markdown]
# The calculated utilities are:

# %%
print('\n'.join([str(k)+':'+str(v) for k, v in TDagent.U.items()]))

# %% [markdown]
# When comparing to previous agents, the result of `PassiveTDAgent` is closer to `PassiveADPAgent`.

# %%
