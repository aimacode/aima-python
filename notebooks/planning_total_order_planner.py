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
# ### TOTAL ORDER PLANNER
#
# In mathematical terminology, **total order**, **linear order** or **simple order** refers to a set *X* which is said to be totally ordered under &le; if the following statements hold for all *a*, *b* and *c* in *X*:
# <br>
# If *a* &le; *b* and *b* &le; *a*, then *a* = *b* (antisymmetry).
# <br>
# If *a* &le; *b* and *b* &le; *c*, then *a* &le; *c* (transitivity).
# <br>
# *a* &le; *b* or *b* &le; *a* (connex relation).
#
# <br>
# In simpler terms, a total order plan is a linear ordering of actions to be taken to reach the goal state.
# There may be several different total-order plans for a particular goal depending on the problem.
# <br>
# <br>
# In the module, the `Linearize` class solves problems using this paradigm.
# At its core, the `Linearize` uses a solved planning graph from `GraphPlan` and finds a valid total-order solution for it.
# Let's have a look at the class.

# %%
from aima.planning import *
from aima.notebook_utils import psource

# %%
psource(Linearize)

# %% [markdown]
# The `filter` method removes the persistence actions (if any) from the planning graph representation.
# <br>
# The `orderlevel` method finds a valid total-ordering of a specified level of the planning-graph, given the state of the graph after the previous level.
# <br>
# The `execute` method sequentially calls `orderlevel` for all the levels in the planning-graph and returns the final total-order solution.
# <br>
# <br>
# Let's look at some examples.

# %%
# total-order solution for air_cargo problem
Linearize(air_cargo()).execute()

# %%
# total-order solution for spare_tire problem
Linearize(spare_tire()).execute()

# %%
# total-order solution for three_block_tower problem
Linearize(three_block_tower()).execute()

# %%
# total-order solution for simple_blocks_world problem
Linearize(simple_blocks_world()).execute()

# %%
# total-order solution for socks_and_shoes problem
Linearize(socks_and_shoes()).execute()
