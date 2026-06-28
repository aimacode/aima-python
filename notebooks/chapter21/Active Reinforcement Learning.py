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
# # ACTIVE REINFORCEMENT LEARNING
#
# This notebook mainly focuses on active reinforce learning algorithms. For a general introduction to reinforcement learning and passive algorithms, please refer to the notebook of **[Passive Reinforcement Learning](./Passive%20Reinforcement%20Learning.ipynb)**.
#
# Unlike Passive Reinforcement Learning in Active Reinforcement Learning, we are not bound by a policy pi and we need to select our actions. In other words, the agent needs to learn an optimal policy. The fundamental tradeoff the agent needs to face is that of exploration vs. exploitation. 
#
# ## QLearning Agent
#
# The QLearningAgent class in the rl module implements the Agent Program described in **Fig 21.8** of the AIMA Book. In Q-Learning the agent learns an action-value function Q which gives the utility of taking a given action in a particular state. Q-Learning does not require a transition model and hence is a model-free method. Let us look into the source before we see some usage examples.

# %%
# %psource QLearningAgent

# %% [markdown]
# The Agent Program can be obtained by creating the instance of the class by passing the appropriate parameters. Because of the __ call __ method the object that is created behaves like a callable and returns an appropriate action as most Agent Programs do. To instantiate the object we need a `mdp` object similar to the `PassiveTDAgent`.
#
#  Let us use the same `GridMDP` object we used above. **Figure 17.1 (sequential_decision_environment)** is similar to **Figure 21.1** but has some discounting parameter as **gamma = 0.9**. The enviroment also implements an exploration function **f** which returns fixed **Rplus** until agent has visited state, action **Ne** number of times. The method **actions_in_state** returns actions possible in given state. It is useful when applying max and argmax operations.

# %% [markdown]
# Let us create our object now. We also use the **same alpha** as given in the footnote of the book on **page 769**: $\alpha(n)=60/(59+n)$ We use **Rplus = 2** and **Ne = 5** as defined in the book. The pseudocode can be referred from **Fig 21.7** in the book.

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.reinforcement_learning import *
from aima.mdp import sequential_decision_environment, value_iteration

# %%
q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, 
                         alpha=lambda n: 60./(59+n))

# %% [markdown]
# Now to try out the q_agent we make use of the **run_single_trial** function in rl.py (which was also used above). Let us use **200** iterations.

# %%
for i in range(200):
    run_single_trial(q_agent,sequential_decision_environment)

# %% [markdown]
# Now let us see the Q Values. The keys are state-action pairs. Where different actions correspond according to:
#
# north = (0, 1)  
# south = (0,-1)  
# west = (-1, 0)  
# east = (1, 0)

# %%
q_agent.Q

# %% [markdown]
# The Utility U of each state is related to Q by the following equation.
#
# $$U (s) = max_a Q(s, a)$$
#
# Let us convert the Q Values above into U estimates.
#
#

# %%
U = defaultdict(lambda: -1000.) # Very Large Negative Value for Comparison see below.
for state_action, value in q_agent.Q.items():
    state, action = state_action
    if U[state] < value:
        U[state] = value

# %% [markdown]
# Now we can output the estimated utility values at each state:

# %%
U

# %% [markdown]
# Let us finally compare these estimates to value_iteration results.

# %%
print(value_iteration(sequential_decision_environment))
