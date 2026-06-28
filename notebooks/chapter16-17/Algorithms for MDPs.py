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
# # Algorithms for MDPs
#
# There are multiple different algorithms for solving MDPs. Some solutions such as value iteration, policy iteration, and linear programming are offline solutions that generate exact results. There are also online solutions computing the results by sampling possible features such as Monte Carlo planning.

# %% [markdown]
# ## Value Iteration
#
# When solving an MDP, our ultimate goal is to obtain an optimal policy. We start by looking at Value Iteration and a visualization that should help us understand it better.
#
# We start by calculating Value/Utility for each of the states. The Value of each state is the expected sum of discounted future rewards given we start in that state and follow a particular policy $\pi$. The value or the utility of a state is given by
#
# $$U(s)=R(s)+\gamma\max_{a\epsilon A(s)}\sum_{s'} P(s'\ |\ s,a)U(s')$$
#
# This is called the Bellman equation. The algorithm Value Iteration (**Fig. 16.2** in the book) relies on finding solutions to this Equation. The intuition Value Iteration works are because values propagate through the state space utilizing local updates. This point will we more clear after we encounter the visualization. 

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.mdp import *
from aima.notebook_utils import psource, pseudocode, plot_pomdp_utility

# %%
psource(value_iteration)

# %% [markdown]
# It takes as inputs two parameters, an MDP to solve and epsilon, the maximum error allowed in the utility of any state. It returns a dictionary containing utilities where the keys are the states and values represent utilities. <br> Value Iteration starts with arbitrary initial values for the utilities, calculates the right side of the Bellman equation and plugs it into the left-hand side, thereby updating the utility of each state from the utilities of its neighbors. 
# This is repeated until equilibrium is reached. 
# It works on the principle of _Dynamic Programming_ - using precomputed information to simplify the subsequent computation. 
# If $U_i(s)$ is the utility value for state $s$ at the $i$ th iteration, the iteration step, called Bellman update, looks like this:
#
# $$ U_{i+1}(s) \leftarrow R(s) + \gamma \max_{a \epsilon A(s)} \sum_{s'} P(s'\ |\ s,a)U_{i}(s') $$
#
# As you might have noticed, `value_iteration` has an infinite loop. How do we decide when to stop iterating? 
# The concept of _contraction_ successfully explains the convergence of value iteration. 
# Refer to **Section 17.2.3** of the book for a detailed explanation. 
# In the algorithm, we calculate a value $delta$ that measures the difference in the utilities of the current time step and the previous time step. 
#
# $$\delta = \max{(\delta, \begin{vmatrix}U_{i + 1}(s) - U_i(s)\end{vmatrix})}$$
#
# This value of delta decreases as the values of $U_i$ converge.
# We terminate the algorithm if the $\delta$ value is less than a threshold value determined by the hyperparameter _epsilon_.
#
# $$\delta \lt \epsilon \frac{(1 - \gamma)}{\gamma}$$
#
# To summarize, the Bellman update is a _contraction_ by a factor of $gamma$ on the space of utility vectors. 
# Hence, from the properties of contractions in general, it follows that `value_iteration` always converges to a unique solution of the Bellman equations whenever $gamma$ is less than 1.
# We then terminate the algorithm when a reasonable approximation is achieved.
# In practice, it often occurs that the policy $pi$ becomes optimal long before the utility function converges. For the given 4 x 3 environment with $gamma = 0.9$, the policy $pi$ is optimal when $i = 4$ (at the 4th iteration), even though the maximum error in the utility function is still 0.46. This can be clarified from **figure 17.6** in the book. Hence, to increase computational efficiency, we often use another method to solve MDPs called Policy Iteration which we will see in the latter part of this notebook. 
# <br>For now, let us solve the **sequential_decision_environment** GridMDP using `value_iteration`.

# %%
value_iteration(sequential_decision_environment)

# %% [markdown]
# To view the pseudocode for the algorithm:

# %%
pseudocode("Value-Iteration")


# %% [markdown]
# ### Visualization
#
# To illustrate that values propagate out of states let us create a simple visualization. We will be using a modified version of the value_iteration function which will store U over time. We will also remove the parameter epsilon and instead add the number of iterations we want.

# %%
def value_iteration_instru(mdp, iterations=20):
    U_over_time = []
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for _ in range(iterations):
        U = U1.copy()
        for s in mdp.states:
            U1[s] = max(q_value(mdp, s, a, U) for a in mdp.actions(s))
        U_over_time.append(U)
    return U_over_time


# %% [markdown]
# Next, we define a function to create the visualization from the utilities returned by **value_iteration_instru**. The reader need not concern himself with the code that immediately follows as it is the usage of Matplotib with IPython Widgets. If you are interested in reading more about these visit [ipywidgets.readthedocs.io](http://ipywidgets.readthedocs.io)

# %%
columns = 4
rows = 3
U_over_time = value_iteration_instru(sequential_decision_environment)

# %%
# %matplotlib inline
from aima.notebook_utils import make_plot_grid_step_function

plot_grid_step = make_plot_grid_step_function(columns, rows, U_over_time)

# %%
import ipywidgets as widgets
from IPython.display import display
from aima.notebook_utils import make_visualize

iteration_slider = widgets.IntSlider(min=1, max=15, step=1, value=0)
w=widgets.interactive(plot_grid_step,iteration=iteration_slider)
display(w)

visualize_callback = make_visualize(iteration_slider)

visualize_button = widgets.ToggleButton(description = "Visualize", value = False)
time_select = widgets.ToggleButtons(description='Extra Delay:',options=['0', '0.1', '0.2', '0.5', '0.7', '1.0'])
a = widgets.interactive(visualize_callback, visualize=visualize_button, time_step=time_select)
display(a)

# %% [markdown]
# Move the slider above to observe how the utility changes across iterations. It is also possible to move the slider using arrow keys or to jump to the value by directly editing the number with a double click. The **Visualize Button** will automatically animate the slider for you. The **Extra Delay Box** allows you to set time delay in seconds up to one second for each time step. There is also an interactive editor for grid-world problems `grid_mdp.py` in the GUI folder for you to play around with.

# %% [markdown]
# ## Policy Iteration
#
# We have already seen that value iteration converges to the optimal policy long before it accurately estimates the utility function. 
# If one action is clearly better than all the others, then the exact magnitude of the utilities in the states involved need not be precise. 
# The policy iteration algorithm works on this insight. 
# The algorithm executes two fundamental steps:
# * **Policy evaluation**: Given a policy _&#960;&#7522;_, calculate _U&#7522; = U(&#960;&#7522;)_, the utility of each state if _&#960;&#7522;_ were to be executed.
# * **Policy improvement**: Calculate a new policy _&#960;&#7522;&#8330;&#8321;_ using one-step look-ahead based on the utility values calculated.
#
# The algorithm terminates when the policy improvement step yields no change in the utilities. 
# We now have a simplified version of the Bellman equation
#
# $$U_i(s) = R(s) + \gamma \sum_{s'}P(s'\ |\ s, \pi_i(s))U_i(s')$$
#
# An important observation in this equation is that this equation doesn't have the `max` operator, which makes it linear.
# For _n_ states, we have _n_ linear equations with _n_ unknowns, which can be solved exactly in time _**O(n&#179;)**_.
# For more implementational details, have a look at **Section 16.2**.
# Let us now look at how the expected utility is found and how `policy_iteration` is implemented.

# %%
psource(expected_utility)

# %%
psource(policy_iteration)

# %% [markdown]
# <br>Fortunately, it is not necessary to do _exact_ policy evaluation. 
# The utilities can instead be reasonably approximated by performing some number of simplified value iteration steps.
# The simplified Bellman update equation for the process is
#
# $$U_{i+1}(s) \leftarrow R(s) + \gamma\sum_{s'}P(s'\ |\ s,\pi_i(s))U_{i}(s')$$
#
# and this is repeated _k_ times to produce the next utility estimate. This is called _modified policy iteration_.

# %%
psource(policy_evaluation)

# %%
policy_iteration(sequential_decision_environment)
