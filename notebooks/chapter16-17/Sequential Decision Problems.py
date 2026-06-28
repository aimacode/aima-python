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
# # Sequential Decision Problems
#
# Now that we have defined MDPs in `MPDs.ipynb` and the tools required to solve MDPs in `Algorithms for MDPs.ipynb`, let us see how Sequential Decision Problems can be solved step by step and how a few built-in tools in the GridMDP class help us better analyze the problem at hand. 
# As always, we will work with the grid world from **Figure 16.1** from the book.
# <img src="images/grid_mdp.jpg" width=500/>
# <br>This is the environment for our agent.
# We assume for now that the environment is _fully observable_, so that the agent always knows where it is.
# We also assume that the transitions are **Markovian**, that is, the probability of reaching state $s'$ from state $s$ depends only on $s$ and not on the history of earlier states.
# Almost all stochastic decision problems can be reframed as a Markov Decision Process just by tweaking the definition of a _state_ for that particular problem.
# <br>
# However, the actions of our agents in this environment are unreliable. In other words, the motion of our agent is stochastic. 
# <br><br>
# More specifically, the agent may - 
# * move correctly in the intended direction with a probability of _0.8_,  
# * move $90^\circ$ to the right of the intended direction with a probability 0.1
# * move $90^\circ$ to the left of the intended direction with a probability 0.1
# <br><br>
# The agent stays put if it bumps into a wall.
# <img src="images/grid_mdp_agent.jpg" width=200/>

# %% [markdown]
# These properties of the agent are called the transition properties and are hardcoded into the GridMDP class as you can see below.

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.mdp import *
from aima.notebook_utils import psource, pseudocode, plot_pomdp_utility

# %%
psource(GridMDP.T)

# %% [markdown]
# To completely define our task environment, we need to specify the utility function for the agent. 
# This is the function that gives the agent a rough estimate of how good being in a particular state is, or how much _reward_ an agent receives by being in that state.
# The agent then tries to maximize the reward it gets.
# As the decision problem is sequential, the utility function will depend on a sequence of states rather than on a single state.
# For now, we simply stipulate that in each state $s$, the agent receives a finite reward $R(s)$.
#
# For any given state, the actions the agent can take are encoded as given below:
# - Move Up: (0, 1)
# - Move Down: (0, -1)
# - Move Left: (-1, 0)
# - Move Right: (1, 0)
# - Do nothing: `None`
#
# We now wonder what a valid solution to the problem might look like. 
# We cannot have fixed action sequences as the environment is stochastic and we can eventually end up in an undesirable state.
# Therefore, a solution must specify what the agent should do for _any_ state the agent might reach.
# <br>
# Such a solution is known as a **policy** and is usually denoted by $\pi$.
# <br>
# The **optimal policy** is the policy that yields the highest expected utility an is usually denoted by $\pi^*$.
# <br>
# The `GridMDP` class has a useful method `to_arrows` that outputs a grid showing the direction the agent should move, given a policy.
# We will use this later to better understand the properties of the environment.

# %%
psource(GridMDP.to_arrows)

# %% [markdown]
# This method directly encodes the actions that the agent can take (described above) to characters representing arrows and shows it in a grid format for human visalization purposes. 
# It converts the received policy from a `dictionary` to a grid using the `to_grid` method.

# %%
psource(GridMDP.to_grid)

# %% [markdown]
# Now that we have all the tools required and a good understanding of the agent and the environment, we consider some cases and see how the agent should behave for each case.

# %% [markdown]
# ## Case 1
# ---
# R(s) = -0.04 in all states except terminal states

# %%
# Note that this environment is also initialized in mdp.py by default
sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

# %% [markdown]
# We will use the `best_policy` function to find the best policy for this environment.
# But, as you can see, `best_policy` requires a utility function as well.
# We already know that the utility function can be found by `value_iteration`.
# Hence, our best policy is:

# %%
pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .001))

# %% [markdown]
# We can now use the `to_arrows` method to see how our agent should pick its actions in the environment.

# %%
from aima.utils import print_table
print_table(sequential_decision_environment.to_arrows(pi))

# %% [markdown]
# This is exactly the output we expected
# <br>
# <img src="images/-0.04.jpg" width=300/>
# <br>
#
# Notice that, because the cost of taking a step is fairly small compared with the penalty for ending up in `(4, 2)` by accident, the optimal policy is conservative. 
# In state `(3, 1)` it recommends taking the long way round, rather than taking the shorter way and risking getting a large negative reward of -1 in `(4, 2)`.

# %% [markdown]
# ## Case 2
# ---
# R(s) = -0.4 in all states except in terminal states

# %%
sequential_decision_environment = GridMDP([[-0.4, -0.4, -0.4, +1],
                                           [-0.4, None, -0.4, -1],
                                           [-0.4, -0.4, -0.4, -0.4]],
                                          terminals=[(3, 2), (3, 1)])

# %%
pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .001))
from aima.utils import print_table
print_table(sequential_decision_environment.to_arrows(pi))

# %% [markdown]
# This is exactly the output we expected
# <img src="images/-0.4.jpg" width=300/>
#
# As the reward for each state is now more negative, life is certainly more unpleasant.
# The agent takes the shortest route to the +1 state and is willing to risk falling into the -1 state by accident.

# %% [markdown]
# ## Case 3
# ---
# R(s) = -4 in all states except terminal states

# %%
sequential_decision_environment = GridMDP([[-4, -4, -4, +1],
                                           [-4, None, -4, -1],
                                           [-4, -4, -4, -4]],
                                          terminals=[(3, 2), (3, 1)])

# %%
pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .001))
from aima.utils import print_table
print_table(sequential_decision_environment.to_arrows(pi))

# %% [markdown]
# This is exactly the output we expected
# <img src="images/-4.jpg" width=300/>
#
# The living reward for each state is now lower than the least rewarding terminal. Life is so _painful_ that the agent heads for the nearest exit as even the worst exit is less painful than any living state.

# %% [markdown]
# ## Case 4
# ---
# R(s) = 4 in all states except terminal states

# %%
sequential_decision_environment = GridMDP([[4, 4, 4, +1],
                                           [4, None, 4, -1],
                                           [4, 4, 4, 4]],
                                          terminals=[(3, 2), (3, 1)])

# %%
pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .001))
from aima.utils import print_table
print_table(sequential_decision_environment.to_arrows(pi))

# %% [markdown]
# In this case, the output we expect is
# <img src="images/4.jpg" width=300/>
# <br>
# As life is positively enjoyable and the agent avoids _both_ exits.
# Even though the output we get is not exactly what we want, it is definitely not wrong.
# The scenario here requires the agent to anything but reach a terminal state, as this is the only way the agent can maximize its reward (total reward tends to infinity), and the program does just that.
# <br>
# Currently, the GridMDP class doesn't support an explicit marker for a "do whatever you like" action or a "don't care" condition.
# You can, however, extend the class to do so.
# <br>
# For in-depth knowledge about sequential decision problems, refer **Section 17.1** in the AIMA book.

# %%
