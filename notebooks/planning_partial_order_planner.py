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
# ### PARTIAL ORDER  PLANNER
# A partial-order planning algorithm is significantly different from a total-order planner.
# The way a partial-order plan works enables it to take advantage of _problem decomposition_ and work on each subproblem separately.
# It works on several subgoals independently, solves them with several subplans, and then combines the plan.
# <br>
# A partial-order planner also follows the **least commitment** strategy, where it delays making choices for as long as possible.
# Variables are not bound unless it is absolutely necessary and new actions are chosen only if the existing actions cannot fulfil the required precondition.
# <br>
# Any planning algorithm that can place two actions into a plan without specifying which comes first is called a **partial-order planner**.
# A partial-order planner searches through the space of plans rather than the space of states, which makes it perform better for certain problems.
# <br>
# <br>
# Let's have a look at the `PartialOrderPlanner` class.

# %%
from aima.planning import *
from aima.notebook_utils import psource

# %%
psource(PartialOrderPlanner)

# %% [markdown]
# We will first describe the data-structures and helper methods used, followed by the algorithm used to find a partial-order plan.

# %% [markdown]
# Each plan has the following four components:
#
# 1. **`actions`**: a set of actions that make up the steps of the plan.
# `actions` is always a subset of `pddl.actions` the set of possible actions for the given planning problem. 
# The `start` and `finish` actions are dummy actions defined to bring uniformity to the problem. The `start` action has no preconditions and its effects constitute the initial state of the planning problem. 
# The `finish` action has no effects and its preconditions constitute the goal state of the planning problem.
# The empty plan consists of just these two dummy actions.
# 2. **`constraints`**: a set of temporal constraints that define the order of performing the actions relative to each other.
# `constraints` does not define a linear ordering, rather it usually represents a directed graph which is also acyclic if the plan is consistent.
# Each ordering is of the form A &lt; B, which reads as "A before B" and means that action A _must_ be executed sometime before action B, but not necessarily immediately before.
# `constraints` stores these as a set of tuples `(Action(A), Action(B))` which is interpreted as given above.
# A constraint cannot be added to `constraints` if it breaks the acyclicity of the existing graph.
# 3. **`causal_links`**: a set of causal-links. 
# A causal link between two actions _A_ and _B_ in the plan is written as _A_ --_p_--> _B_ and is read as "A achieves p for B".
# This imples that _p_ is an effect of _A_ and a precondition of _B_.
# It also asserts that _p_ must remain true from the time of action _A_ to the time of action _B_.
# Any violation of this rule is called a threat and must be resolved immediately by adding suitable ordering constraints.
# `causal_links` stores this information as tuples `(Action(A), precondition(p), Action(B))` which is interpreted as given above.
# Causal-links can also be called **protection-intervals**, because the link _A_ --_p_--> _B_ protects _p_ from being negated over the interval from _A_ to _B_.
# 4. **`agenda`**: a set of open-preconditions.
# A precondition is open if it is not achieved by some action in the plan.
# Planners will work to reduce the set of open preconditions to the empty set, without introducing a contradiction.
# `agenda` stored this information as tuples `(precondition(p), Action(A))` where p is a precondition of the action A.
#
# A **consistent plan** is a plan in which there are no cycles in the ordering constraints and no conflicts with the causal-links.
# A consistent plan with no open preconditions is a **solution**.
# <br>
# <br>
# Let's briefly glance over the helper functions before going into the actual algorithm.
# <br>
# **`expand_actions`**: generates all possible actions with variable bindings for use as a heuristic of selection of an open precondition.
# <br>
# **`find_open_precondition`**: finds a precondition from the agenda with the least number of actions that fulfil that precondition.
# This heuristic helps form mandatory ordering constraints and causal-links to further simplify the problem and reduce the probability of encountering a threat.
# <br>
# **`find_action_for_precondition`**: finds an action that fulfils the given precondition along with the absolutely necessary variable bindings in accordance with the principle of _least commitment_.
# In case of multiple possible actions, the action with the least number of effects is chosen to minimize the chances of encountering a threat.
# <br>
# **`cyclic`**: checks if a directed graph is cyclic.
# <br>
# **`add_const`**: adds `constraint` to `constraints` if the newly formed graph is acyclic and returns `constraints` otherwise.
# <br>
# **`is_a_threat`**: checks if the given `effect` negates the given `precondition`.
# <br>
# **`protect`**: checks if the given `action` poses a threat to the given `causal_link`.
# If so, the threat is resolved by either promotion or demotion, whichever generates acyclic temporal constraints.
# If neither promotion or demotion work, the chosen action is not the correct fit or the planning problem cannot be solved altogether.
# <br>
# **`convert`**: converts a graph from a list of edges to an `Action` : `set` mapping, for use in topological sorting.
# <br>
# **`toposort`**: a generator function that generates a topological ordering of a given graph as a list of sets.
# Each set contains an action or several actions.
# If a set has more that one action in it, it means that permutations between those actions also produce a valid plan.
# <br>
# **`display_plan`**: displays the `causal_links`, `constraints` and the partial order plan generated from `toposort`.
# <br>

# %% [markdown]
# The **`execute`** method executes the algorithm, which is summarized below:
# <br>
# 1. An open precondition is selected (a sub-goal that we want to achieve).
# 2. An action that fulfils the open precondition is chosen.
# 3. Temporal constraints are updated.
# 4. Existing causal links are protected. Protection is a method that checks if the causal links conflict
#    and if they do, temporal constraints are added to fix the threats.
# 5. The set of open preconditions is updated.
# 6. Temporal constraints of the selected action and the next action are established.
# 7. A new causal link is added between the selected action and the owner of the open precondition.
# 8. The set of new causal links is checked for threats and if found, the threat is removed by either promotion or demotion.
#    If promotion or demotion is unable to solve the problem, the planning problem cannot be solved with the current sequence of actions
#    or it may not be solvable at all.
# 9. These steps are repeated until the set of open preconditions is empty.

# %% [markdown]
# A partial-order plan can be used to generate different valid total-order plans.
# This step is called **linearization** of the partial-order plan.
# All possible linearizations of a partial-order plan for `socks_and_shoes` looks like this.
# <br>
# ![title](images/pop.jpg)
# <br>
# Linearization can be carried out in many ways, but the most efficient way is to represent the set of temporal constraints as a directed graph.
# We can easily realize that the graph should also be acyclic as cycles in constraints means that the constraints are inconsistent.
# This acyclicity is enforced by the `add_const` method, which adds a new constraint only if the acyclicity of the existing graph is not violated.
# The `protect` method also checks for acyclicity of the newly-added temporal constraints to make a decision between promotion and demotion in case of a threat.
# This property of a graph created from the temporal constraints of a valid partial-order plan allows us to use topological sort to order the constraints linearly.
# A topological sort may produce several different valid solutions for a given directed acyclic graph.

# %% [markdown]
# Now that we know how `PartialOrderPlanner` works, let's solve a few problems using it.

# %%
st = spare_tire()
pop = PartialOrderPlanner(st)
pop.execute()

# %% [markdown]
# We observe that in the given partial order plan, Remove(Flat, Axle) and Remove(Spare, Trunk) are in the same set.
# This means that the order of performing these actions does not affect the final outcome.
# That aside, we also see that the PutOn(Spare, Axle) action has to be performed after both the Remove actions are complete, which seems logically consistent.

# %%
sbw = simple_blocks_world()
pop = PartialOrderPlanner(sbw)
pop.execute()

# %% [markdown]
# We see that this plan does not have flexibility in selecting actions, ie, actions should be performed in this order and this order only, to successfully reach the goal state.

# %%
ss = socks_and_shoes()
pop = PartialOrderPlanner(ss)
pop.execute()

# %% [markdown]
# This plan again doesn't have constraints in selecting socks or shoes.
# As long as both socks are worn before both shoes, we are fine.
# Notice however, there is one valid solution,
# <br>
# LeftSock -> LeftShoe -> RightSock -> RightShoe
# <br>
# that the algorithm could not find as it cannot be represented as a general partially-ordered plan but is a specific total-order solution.

# %% [markdown]
# ### Runtime differences
# Let's briefly take a look at the running time of all the three algorithms on the `socks_and_shoes` problem.

# %%
ss = socks_and_shoes()

# %%
# %%timeit
GraphPlan(ss).execute()

# %%
# %%timeit
Linearize(ss).execute()

# %%
# %%timeit
PartialOrderPlanner(ss).execute(display=False)

# %% [markdown]
# We observe that `GraphPlan` is about 4 times faster than `Linearize` because `Linearize` essentially runs a `GraphPlan` subroutine under the hood and then carries out some transformations on the solved planning-graph.
# <br>
# We also find that `GraphPlan` is slightly faster than `PartialOrderPlanner`, but this is mainly due to the `expand_actions` method in `PartialOrderPlanner` that slows it down as it generates all possible permutations of actions and variable bindings.
# <br>
# Without heuristic functions, `PartialOrderPlanner` will be atleast as fast as `GraphPlan`, if not faster, but will have a higher tendency to encounter threats and conflicts which might take additional time to resolve.
# <br>
# Different planning algorithms work differently for different problems.
