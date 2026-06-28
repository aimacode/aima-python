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
# ## SOLVING PLANNING PROBLEMS
# ----
# ### GRAPHPLAN
# <br>
# The GraphPlan algorithm is a popular method of solving classical planning problems.
# Before we get into the details of the algorithm, let's look at a special data structure called **planning graph**, used to give better heuristic estimates and plays a key role in the GraphPlan algorithm.

# %% [markdown]
# ### Planning Graph
# A planning graph is a directed graph organized into levels. 
# Each level contains information about the current state of the knowledge base and the possible state-action links to and from that level.
# The first level contains the initial state with nodes representing each fluent that holds in that level.
# This level has state-action links linking each state to valid actions in that state.
# Each action is linked to all its preconditions and its effect states.
# Based on these effects, the next level is constructed.
# The next level contains similarly structured information about the next state.
# In this way, the graph is expanded using state-action links till we reach a state where all the required goals hold true simultaneously.
# We can say that we have reached our goal if none of the goal states in the current level are mutually exclusive.
# This will be explained in detail later.
# <br>
# Planning graphs only work for propositional planning problems, hence we need to eliminate all variables by generating all possible substitutions.
# <br>
# For example, the planning graph of the `have_cake_and_eat_cake_too` problem might look like this
# ![title](images/cake_graph.jpg)
# <br>
# The black lines indicate links between states and actions.
# <br>
# In every planning problem, we are allowed to carry out the `no-op` action, ie, we can choose no action for a particular state.
# These are called 'Persistence' actions and are represented in the graph by the small square boxes.
# In technical terms, a persistence action has effects same as its preconditions.
# This enables us to carry a state to the next level.
# <br>
# <br>
# The gray lines indicate mutual exclusivity.
# This means that the actions connected bya gray line cannot be taken together.
# Mutual exclusivity (mutex) occurs in the following cases:
# 1. **Inconsistent effects**: One action negates the effect of the other. For example, _Eat(Cake)_ and the persistence of _Have(Cake)_ have inconsistent effects because they disagree on the effect _Have(Cake)_
# 2. **Interference**: One of the effects of an action is the negation of a precondition of the other. For example, _Eat(Cake)_ interferes with the persistence of _Have(Cake)_ by negating its precondition.
# 3. **Competing needs**: One of the preconditions of one action is mutually exclusive with a precondition of the other. For example, _Bake(Cake)_ and _Eat(Cake)_ are mutex because they compete on the value of the _Have(Cake)_ precondition.

# %% [markdown]
# In the module, planning graphs have been implemented using two classes, `Level` which stores data for a particular level and `Graph` which connects multiple levels together.
# Let's look at the `Level` class.

# %%
from aima.planning import *
from aima.notebook_utils import psource

# %%
psource(Level)

# %% [markdown]
# Each level stores the following data
# 1. The current state of the level in `current_state`
# 2. Links from an action to its preconditions in `current_action_links`
# 3. Links from a state to the possible actions in that state in `current_state_links`
# 4. Links from each action to its effects in `next_action_links`
# 5. Links from each possible next state from each action in `next_state_links`. This stores the same information as the `current_action_links` of the next level.
# 6. Mutex links in `mutex`.
# <br>
# <br>
# The `find_mutex` method finds the mutex links according to the points given above.
# <br>
# The `build` method populates the data structures storing the state and action information.
# Persistence actions for each clause in the current state are also defined here. 
# The newly created persistence action has the same name as its state, prefixed with a 'P'.

# %% [markdown]
# Let's now look at the `Graph` class.

# %%
psource(Graph)

# %% [markdown]
# The class stores a problem definition in `pddl`, 
# a knowledge base in `kb`, 
# a list of `Level` objects in `levels` and 
# all the possible arguments found in the initial state of the problem in `objects`.
# <br>
# The `expand_graph` method generates a new level of the graph.
# This method is invoked when the goal conditions haven't been met in the current level or the actions that lead to it are mutually exclusive.
# The `non_mutex_goals` method checks whether the goals in the current state are mutually exclusive.
# <br>
# <br>
# Using these two classes, we can define a planning graph which can either be used to provide reliable heuristics for planning problems or used in the `GraphPlan` algorithm.
# <br>
# Let's have a look at the `GraphPlan` class.

# %%
psource(GraphPlan)

# %% [markdown]
# Given a planning problem defined as a PlanningProblem, `GraphPlan` creates a planning graph stored in `graph` and expands it till it reaches a state where all its required goals are present simultaneously without mutual exclusivity.
# <br>
# Once a goal is found, `extract_solution` is called.
# This method recursively finds the path to a solution given a planning graph.
# In the case where `extract_solution` fails to find a solution for a set of goals as a given level, we record the `(level, goals)` pair as a **no-good**.
# Whenever `extract_solution` is called again with the same level and goals, we can find the recorded no-good and immediately return failure rather than searching again. 
# No-goods are also used in the termination test.
# <br>
# The `check_leveloff` method checks if the planning graph for the problem has **levelled-off**, ie, it has the same states, actions and mutex pairs as the previous level.
# If the graph has already levelled off and we haven't found a solution, there is no point expanding the graph, as it won't lead to anything new.
# In such a case, we can declare that the planning problem is unsolvable with the given constraints.
# <br>
# <br>
# To summarize, the `GraphPlan` algorithm calls `expand_graph` and tests whether it has reached the goal and if the goals are non-mutex.
# <br>
# If so, `extract_solution` is invoked which recursively reconstructs the solution from the planning graph.
# <br>
# If not, then we check if our graph has levelled off and continue if it hasn't.

# %% [markdown]
# Let's solve a few planning problems that we had defined earlier.

# %% [markdown]
# #### Air cargo problem
# In accordance with the summary above, we have defined a helper function to carry out `GraphPlan` on the `air_cargo` problem.
# The function is pretty straightforward.
# Let's have a look.

# %%
psource(air_cargo_graph_plan)

# %% [markdown]
# Let's instantiate the problem and find a solution using this helper function.

# %%
airCargoG = air_cargo_graph_plan()
airCargoG

# %% [markdown]
# Each element in the solution is a valid action.
# The solution is separated into lists for each level.
# The actions prefixed with a 'P' are persistence actions and can be ignored.
# They simply carry certain states forward.
# We have another helper function `linearize` that presents the solution in a more readable format, much like a total-order planner, but it is _not_ a total-order planner.

# %%
linearize(airCargoG)

# %% [markdown]
# Indeed, this is a correct solution.
# <br>
# There are similar helper functions for some other planning problems.
# <br>
# Lets' try solving the spare tire problem.

# %%
spareTireG = spare_tire_graph_plan()
linearize(spareTireG)

# %% [markdown]
# Solution for the cake problem

# %%
cakeProblemG = have_cake_and_eat_cake_too_graph_plan()
linearize(cakeProblemG)

# %% [markdown]
# Solution for the Sussman's Anomaly configuration of three blocks.

# %%
sussmanAnomalyG = three_block_tower_graph_plan()
linearize(sussmanAnomalyG)

# %% [markdown]
# Solution of the socks and shoes problem

# %%
socksShoesG = socks_and_shoes_graph_plan()
linearize(socksShoesG)
