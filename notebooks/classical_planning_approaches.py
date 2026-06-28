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
# # Classical Planning
# ---
# # Classical Planning Approaches
#
# ## Introduction 
# ***Planning*** combines the two major areas of AI: *search* and *logic*. A planner can be seen either as a program that searches for a solution or as one that constructively proves the existence of a solution.
#
# Currently, the most popular and effective approaches to fully automated planning are:
# - searching using a *planning graph*;
# - *state-space search* with heuristics;
# - translating to a *constraint satisfaction (CSP) problem*;
# - translating to a *boolean satisfiability (SAT) problem*.

# %%
from aima.planning import *

# %% [markdown]
# ## Planning as Planning Graph Search
#
# A *planning graph* is a directed graph organized into levels each of which contains information about the current state of the knowledge base and the possible state-action links to and from that level. 
#
# The first level contains the initial state with nodes representing each fluent that holds in that level. This level has state-action links linking each state to valid actions in that state. Each action is linked to all its preconditions and its effect states. Based on these effects, the next level is constructed and contains similarly structured information about the next state. In this way, the graph is expanded using state-action links till we reach a state where all the required goals hold true simultaneously.
#
# In every planning problem, we are allowed to carry out the *no-op* action, ie, we can choose no action for a particular state. These are called persistence actions and has effects same as its preconditions. This enables us to carry a state to the next level.
#
# Mutual exclusivity (*mutex*) between two actions means that these cannot be taken together and occurs in the following cases:
# - *inconsistent effects*: one action negates the effect of the other;
# - *interference*: one of the effects of an action is the negation of a precondition of the other;
# - *competing needs*: one of the preconditions of one action is mutually exclusive with a precondition of the other.
#
# We can say that we have reached our goal if none of the goal states in the current level are mutually exclusive.

# %%
# %psource Graph

# %%
# %psource Level

# %% [markdown]
# A *planning graph* can be used to give better heuristic estimates which can be applied to any of the search techniques. Alternatively, we can search for a solution over the space formed by the planning graph, using an algorithm called `GraphPlan`.
#
# The `GraphPlan` algorithm repeatedly adds a level to a planning graph. Once all the goals show up as non-mutex in the graph, the algorithm runs backward from the last level to the first searching for a plan that solves the problem. If that fails, it records the (level , goals) pair as a *no-good* (as in constraint learning for CSPs), expands another level and tries again, terminating with failure when there is no reason to go on. 

# %%
# %psource GraphPlan

# %% [markdown]
# ## Planning as State-Space Search

# %% [markdown]
# The description of a planning problem defines a search problem: we can search from the initial state through the space of states, looking for a goal. One of the nice advantages of the declarative representation of action schemas is that we can also search backward from the goal, looking for the initial state. 
#
# However, neither forward nor backward search is efficient without a good heuristic function because the real-world planning problems often have large state spaces. A heuristic function $h(s)$ estimates the distance from a state $s$ to the goal and, if it is admissible, ie if does not overestimate, then we can use $A^∗$ search to find optimal solutions.
#
# Planning uses a factored representation for states and action schemas which makes it possible to define good domain-independent heuristics to prune the search space.
#
# An admissible heuristic can be derived by defining a relaxed problem that is easier to solve. The length of the solution of this easier problem then becomes the heuristic for the original problem. Assume that all goals and preconditions contain only positive literals, ie that the problem is defined according to the *Stanford Research Institute Problem Solver* (STRIPS) notation: we want to create a relaxed version of the original problem that will be easier to solve by ignoring delete lists from all actions, ie removing all negative literals from effects. As shown in <a name="ref-1"/>[[1]](#cite-hoffmann2001ff) the planning graph of a relaxed problem does not contain any mutex relations at all (which is the crucial thing when building a planning graph) and for this reason GraphPlan will never backtrack looking for a solution: for this reason the **ignore delete lists** heuristic makes it possible to find the optimal solution for relaxed problem in polynomial time through `GraphPlan` algorithm.

# %%
from aima.search import *

# %% [markdown]
# ### Forward State-Space Search

# %% [markdown]
# Forward search through the space of states, starting in the initial state and using the problem’s actions to search forward for a member of the set of goal states.

# %%
# %psource ForwardPlan

# %% [markdown]
# ### Backward Relevant-States Search

# %% [markdown]
# Backward search through sets of relevant states, starting at the set of states representing the goal and using the inverse of the actions to search backward for the initial state.

# %%
# %psource BackwardPlan

# %% [markdown]
# ## Planning as Constraint Satisfaction Problem

# %% [markdown]
# In forward planning, the search is constrained by the initial state and only uses the goal as a stopping criterion and as a source for heuristics. In regression planning, the search is constrained by the goal and only uses the start state as a stopping criterion and as a source for heuristics. By converting the problem to a constraint satisfaction problem (CSP), the initial state can be used to prune what is not reachable and the goal to prune what is not useful. The CSP will be defined for a finite number of steps; the number of steps can be adjusted to find the shortest plan. One of the CSP methods can then be used to solve the CSP and thus find a plan.
#
# To construct a CSP from a planning problem, first choose a fixed planning *horizon*, which is the number of time steps over which to plan. Suppose the horizon is 
# $k$. The CSP has the following variables:
#
# - a *state variable* for each feature and each time from 0 to $k$. If there are $n$ features for a horizon of $k$, there are $n \cdot (k+1)$ state variables. The domain of the state variable is the domain of the corresponding feature;
# - an *action variable*, $Action_t$, for each $t$ in the range 0 to $k-1$. The domain of $Action_t$, represents the action that takes the agent from the state at time $t$ to the state at time $t+1$.
#
# There are several types of constraints:
#
# - a *precondition constraint* between a state variable at time $t$ and the variable $Actiont_t$ constrains what actions are legal at time $t$;
# - an *effect constraint* between $Action_t$ and a state variable at time $t+1$ constrains the values of a state variable that is a direct effect of the action;
# - a *frame constraint* among a state variable at time $t$, the variable $Action_t$, and the corresponding state variable at time $t+1$ specifies when the variable that does not change as a result of an action has the same value before and after the action;
# - an *initial-state constraint* constrains a variable on the initial state (at time 0). The initial state is represented as a set of domain constraints on the state variables at time 0;
# - a *goal constraint* constrains the final state to be a state that satisfies the achievement goal. These are domain constraints on the variables that appear in the goal;
# - a *state constraint* is a constraint among variables at the same time step. These can include physical constraints on the state or can ensure that states that violate maintenance goals are forbidden. This is extra knowledge beyond the power of the feature-based or PDDL representations of the action.
#
# The PDDL representation gives precondition, effect and frame constraints for each time 
# $t$ as follows:
#
# - for each $Var = v$ in the precondition of action $A$, there is a precondition constraint:
# $$ Var_t = v \leftarrow Action_t = A $$
# that specifies that if the action is to be $A$, $Var_t$ must have value $v$ immediately before. This constraint is violated when $Action_t = A$ and $Var_t \neq v$, and thus is equivalent to $\lnot{(Var_t \neq v \land Action_t = A)}$;
# - or each $Var = v$ in the effect of action $A$, there is a effect constraint:
# $$ Var_{t+1} = v \leftarrow Action_t = A $$
# which is violated when $Action_t = A$ and $Var_{t+1} \neq v$, and thus is equivalent to $\lnot{(Var_{t+1} \neq v \land Action_t = A)}$;
# - for each $Var$, there is a frame constraint, where $As$ is the set of actions that include $Var$ in the effect of the action:
# $$ Var_{t+1} = Var_t \leftarrow Action_t \notin As $$
# which specifies that the feature $Var$ has the same value before and after any action that does not affect $Var$.
#
# The CSP representation assumes a fixed planning horizon (ie a fixed number of steps). To find a plan over any number of steps, the algorithm can be run for a horizon of $k = 0, 1, 2, \dots$ until a solution is found.

# %%
from aima.csp import *

# %%
# %psource CSPlan

# %% [markdown]
# ## Planning as Boolean Satisfiability Problem
#
# As shown in <a name="ref-2"/>[[2]](cite-kautz1992planning) the translation of a *Planning Domain Definition Language* (PDDL) description into a *Conjunctive Normal Form* (CNF) formula is a series of straightforward steps:
# - *propositionalize the actions*: replace each action schema with a set of ground actions formed by substituting constants for each of the variables. These ground actions are not part of the translation, but will be used in subsequent steps;
# - *define the initial state*: assert $F^0$ for every fluent $F$ in the problem’s initial state, and $\lnot{F}$ for every fluent not mentioned in the initial state;
# - *propositionalize the goal*: for every variable in the goal, replace the literals that contain the variable with a disjunction over constants;
# - *add successor-state axioms*: for each fluent $F$, add an axiom of the form
#
# $$ F^{t+1} \iff ActionCausesF^t \lor (F^t \land \lnot{ActionCausesNotF^t}) $$
#
# where $ActionCausesF$ is a disjunction of all the ground actions that have $F$ in their add list, and $ActionCausesNotF$ is a disjunction of all the ground actions that have $F$ in their delete list;
# - *add precondition axioms*: for each ground action $A$, add the axiom $A^t \implies PRE(A)^t$, that is, if an action is taken at time $t$, then the preconditions must have been true;
# - *add action exclusion axioms*: say that every action is distinct from every other action.
#
# A propositional planning procedure implements the basic idea just given but, because the agent does not know how many steps it will take to reach the goal, the algorithm tries each possible number of steps $t$, up to some maximum conceivable plan length $T_{max}$ . In this way, it is guaranteed to find the shortest plan if one exists. Because of the way the propositional planning procedure searches for a solution, this approach cannot be used in a partially observable environment, ie WalkSAT, but would just set the unobservable variables to the values it needs to create a solution.

# %%
from aima.logic import *

# %%
# %psource SATPlan

# %%
# %psource SAT_plan

# %% [markdown] pycharm={}
# ## Experimental Results

# %% [markdown]
# ### Blocks World

# %%
# %psource three_block_tower

# %% [markdown]
# #### GraphPlan

# %%
# %time blocks_world_solution = GraphPlan(three_block_tower()).execute()
linearize(blocks_world_solution)

# %% [markdown]
# #### ForwardPlan

# %%
# %time blocks_world_solution = uniform_cost_search(ForwardPlan(three_block_tower()), display=True).solution()
blocks_world_solution = list(map(lambda action: Expr(action.name, *action.args), blocks_world_solution))
blocks_world_solution

# %% [markdown]
# #### ForwardPlan with Ignore Delete Lists Heuristic

# %%
# %time blocks_world_solution = astar_search(ForwardPlan(three_block_tower()), display=True).solution()
blocks_world_solution = list(map(lambda action: Expr(action.name, *action.args), blocks_world_solution))
blocks_world_solution

# %% [markdown]
# #### BackwardPlan

# %%
# %time blocks_world_solution = uniform_cost_search(BackwardPlan(three_block_tower()), display=True).solution()
blocks_world_solution = list(map(lambda action: Expr(action.name, *action.args), blocks_world_solution))
blocks_world_solution[::-1]

# %% [markdown]
# #### BackwardPlan with Ignore Delete Lists Heuristic

# %%
# %time blocks_world_solution = astar_search(BackwardPlan(three_block_tower()), display=True).solution()
blocks_world_solution = list(map(lambda action: Expr(action.name, *action.args), blocks_world_solution))
blocks_world_solution[::-1]

# %% [markdown]
# #### CSPlan

# %%
# %time blocks_world_solution = CSPlan(three_block_tower(), 3, arc_heuristic=no_heuristic)
blocks_world_solution

# %% [markdown]
# #### CSPlan with SAT UP Arc Heuristic

# %%
# %time blocks_world_solution = CSPlan(three_block_tower(), 3, arc_heuristic=sat_up)
blocks_world_solution

# %% [markdown]
# #### SATPlan with DPLL

# %%
# %time blocks_world_solution = SATPlan(three_block_tower(), 4, SAT_solver=dpll_satisfiable)
blocks_world_solution

# %% [markdown]
# #### SATPlan with CDCL

# %%
# %time blocks_world_solution = SATPlan(three_block_tower(), 4, SAT_solver=cdcl_satisfiable)
blocks_world_solution

# %% [markdown]
# ### Spare Tire

# %%
# %psource spare_tire

# %% [markdown]
# #### GraphPlan

# %%
# %time spare_tire_solution = GraphPlan(spare_tire()).execute()
linearize(spare_tire_solution)

# %% [markdown]
# #### ForwardPlan

# %%
# %time spare_tire_solution = uniform_cost_search(ForwardPlan(spare_tire()), display=True).solution()
spare_tire_solution = list(map(lambda action: Expr(action.name, *action.args), spare_tire_solution))
spare_tire_solution

# %% [markdown]
# #### ForwardPlan with Ignore Delete Lists Heuristic

# %%
# %time spare_tire_solution = astar_search(ForwardPlan(spare_tire()), display=True).solution()
spare_tire_solution = list(map(lambda action: Expr(action.name, *action.args), spare_tire_solution))
spare_tire_solution

# %% [markdown]
# #### BackwardPlan

# %%
# %time spare_tire_solution = uniform_cost_search(BackwardPlan(spare_tire()), display=True).solution()
spare_tire_solution = list(map(lambda action: Expr(action.name, *action.args), spare_tire_solution))
spare_tire_solution[::-1]

# %% [markdown]
# #### BackwardPlan with Ignore Delete Lists Heuristic

# %%
# %time spare_tire_solution = astar_search(BackwardPlan(spare_tire()), display=True).solution()
spare_tire_solution = list(map(lambda action: Expr(action.name, *action.args), spare_tire_solution))
spare_tire_solution[::-1]

# %% [markdown]
# #### CSPlan

# %%
# %time spare_tire_solution = CSPlan(spare_tire(), 3, arc_heuristic=no_heuristic)
spare_tire_solution

# %% [markdown]
# #### CSPlan with SAT UP Arc Heuristic

# %%
# %time spare_tire_solution = CSPlan(spare_tire(), 3, arc_heuristic=sat_up)
spare_tire_solution

# %% [markdown]
# #### SATPlan with DPLL

# %%
# %time spare_tire_solution = SATPlan(spare_tire(), 4, SAT_solver=dpll_satisfiable)
spare_tire_solution

# %% [markdown]
# #### SATPlan with CDCL

# %%
# %time spare_tire_solution = SATPlan(spare_tire(), 4, SAT_solver=cdcl_satisfiable)
spare_tire_solution

# %% [markdown]
# ### Shopping Problem

# %%
# %psource shopping_problem

# %% [markdown]
# #### GraphPlan

# %%
# %time shopping_problem_solution = GraphPlan(shopping_problem()).execute()
linearize(shopping_problem_solution)

# %% [markdown]
# #### ForwardPlan

# %%
# %time shopping_problem_solution = uniform_cost_search(ForwardPlan(shopping_problem()), display=True).solution()
shopping_problem_solution = list(map(lambda action: Expr(action.name, *action.args), shopping_problem_solution))
shopping_problem_solution

# %% [markdown]
# #### ForwardPlan with Ignore Delete Lists Heuristic

# %%
# %time shopping_problem_solution = astar_search(ForwardPlan(shopping_problem()), display=True).solution()
shopping_problem_solution = list(map(lambda action: Expr(action.name, *action.args), shopping_problem_solution))
shopping_problem_solution

# %% [markdown]
# #### BackwardPlan

# %%
# %time shopping_problem_solution = uniform_cost_search(BackwardPlan(shopping_problem()), display=True).solution()
shopping_problem_solution = list(map(lambda action: Expr(action.name, *action.args), shopping_problem_solution))
shopping_problem_solution[::-1]

# %% [markdown]
# #### BackwardPlan with Ignore Delete Lists Heuristic

# %%
# %time shopping_problem_solution = astar_search(BackwardPlan(shopping_problem()), display=True).solution()
shopping_problem_solution = list(map(lambda action: Expr(action.name, *action.args), shopping_problem_solution))
shopping_problem_solution[::-1]

# %% [markdown]
# #### CSPlan

# %%
# %time shopping_problem_solution = CSPlan(shopping_problem(), 5, arc_heuristic=no_heuristic)
shopping_problem_solution

# %% [markdown]
# #### CSPlan with SAT UP Arc Heuristic

# %%
# %time shopping_problem_solution = CSPlan(shopping_problem(), 5, arc_heuristic=sat_up)
shopping_problem_solution

# %% [markdown]
# #### SATPlan with CDCL

# %%
# %time shopping_problem_solution = SATPlan(shopping_problem(), 5, SAT_solver=cdcl_satisfiable)
shopping_problem_solution

# %% [markdown]
# ### Air Cargo

# %%
# %psource air_cargo

# %% [markdown]
# #### GraphPlan

# %%
# %time air_cargo_solution = GraphPlan(air_cargo()).execute()
linearize(air_cargo_solution)

# %% [markdown]
# #### ForwardPlan

# %%
# %time air_cargo_solution = uniform_cost_search(ForwardPlan(air_cargo()), display=True).solution()
air_cargo_solution = list(map(lambda action: Expr(action.name, *action.args), air_cargo_solution))
air_cargo_solution

# %% [markdown]
# #### ForwardPlan with Ignore Delete Lists Heuristic

# %%
# %time air_cargo_solution = astar_search(ForwardPlan(air_cargo()), display=True).solution()
air_cargo_solution = list(map(lambda action: Expr(action.name, *action.args), air_cargo_solution))
air_cargo_solution

# %% [markdown]
# #### BackwardPlan

# %%
# %time air_cargo_solution = uniform_cost_search(BackwardPlan(air_cargo()), display=True).solution()
air_cargo_solution = list(map(lambda action: Expr(action.name, *action.args), air_cargo_solution))
air_cargo_solution[::-1]

# %% [markdown]
# #### BackwardPlan with Ignore Delete Lists Heuristic

# %%
# %time air_cargo_solution = astar_search(BackwardPlan(air_cargo()), display=True).solution()
air_cargo_solution = list(map(lambda action: Expr(action.name, *action.args), air_cargo_solution))
air_cargo_solution[::-1]

# %% [markdown]
# #### CSPlan

# %%
# %time air_cargo_solution = CSPlan(air_cargo(), 6, arc_heuristic=no_heuristic)
air_cargo_solution

# %% [markdown]
# #### CSPlan with SAT UP Arc Heuristic

# %%
# %time air_cargo_solution = CSPlan(air_cargo(), 6, arc_heuristic=sat_up)
air_cargo_solution

# %% [markdown]
# ## References
#
# <a name="cite-hoffmann2001ff"/><sup>[[1]](#ref-1) </sup>Hoffmann, J&ouml;rg. 2001. _FF: The fast-forward planning system_.
#
# <a name="cite-kautz1992planning"/><sup>[[2]](#ref-2) </sup>Kautz, Henry A and Selman, Bart and others. 1992. _Planning as Satisfiability_.
