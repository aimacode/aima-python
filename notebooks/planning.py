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
# # Planning
# #### Chapters 10-11
# ----

# %% [markdown]
# This notebook serves as supporting material for topics covered in **Chapter 10 - Classical Planning** and **Chapter 11 - Planning and Acting in the Real World** from the book *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu)*. 
# This notebook uses implementations from the [planning.py](https://github.com/aimacode/aima-python/blob/master/planning.py) module. 
# See the [intro notebook](https://github.com/aimacode/aima-python/blob/master/intro.ipynb) for instructions.
#
# We'll start by looking at `PlanningProblem` and `Action` data types for defining problems and actions. 
# Then, we will see how to use them by trying to plan a trip from *Sibiu* to *Bucharest* across the familiar map of Romania, from [search.ipynb](https://github.com/aimacode/aima-python/blob/master/search.ipynb) 
# followed by some common planning problems and methods of solving them.
#
# Let's start by importing everything from the planning module.

# %%
from aima.planning import *
from aima.notebook_utils import psource

# %% [markdown]
# ## CONTENTS
#
# **Classical Planning**
# - PlanningProblem
# - Action
# - Planning Problems
#     * Air cargo problem
#     * Spare tire problem
#     * Three block tower problem
#     * Shopping Problem
#     * Socks and shoes problem
#     * Cake problem
# - Solving Planning Problems
#     * GraphPlan
#     * Linearize
#     * PartialOrderPlanner
# <br>
#
# **Planning in the real world**
# - Problem
# - HLA
# - Planning Problems
#     * Job shop problem
#     * Double tennis problem
# - Solving Planning Problems
#     * Hierarchical Search
#     * Angelic Search

# %% [markdown]
# ## PlanningProblem
#
# PDDL stands for Planning Domain Definition Language.
# The `PlanningProblem` class is used to represent planning problems in this module. The following attributes are essential to be able to define a problem:
# * an initial state
# * a set of goals
# * a set of viable actions that can be executed in the search space of the problem
#
# View the source to see how the Python code tries to realise these.

# %%
psource(PlanningProblem)

# %% [markdown]
# The `init` attribute is an expression that forms the initial knowledge base for the problem.
# <br>
# The `goals` attribute is an expression that indicates the goals to be reached by the problem.
# <br>
# Lastly, `actions` contains a list of `Action` objects that may be executed in the search space of the problem.
# <br>
# The `goal_test` method checks if the goal has been reached.
# <br>
# The `act` method acts out the given action and updates the current state.
# <br>
#

# %% [markdown]
# ## ACTION
#
# To be able to model a planning problem properly, it is essential to be able to represent an Action. Each action we model requires at least three things:
# * preconditions that the action must meet
# * the effects of executing the action
# * some expression that represents the action

# %% [markdown]
# The module models actions using the `Action` class

# %%
psource(Action)

# %% [markdown]
# This class represents an action given the expression, the preconditions and its effects. 
# A list `precond` stores the preconditions of the action and a list `effect` stores its effects.
# Negative preconditions and effects are input using a `~` symbol before the clause, which are internally prefixed with a `Not` to make it easier to work with.
# For example, the negation of `At(obj, loc)` will be input as `~At(obj, loc)` and internally represented as `NotAt(obj, loc)`. 
# This equivalently creates a new clause for each negative literal, removing the hassle of maintaining two separate knowledge bases.
# This greatly simplifies algorithms like `GraphPlan` as we will see later.
# The `convert` method takes an input string, parses it, removes conjunctions if any and returns a list of `Expr` objects.
# The `check_precond` method checks if the preconditions for that action are valid, given a `kb`.
# The `act` method carries out the action on the given knowledge base.

# %% [markdown]
# Now lets try to define a planning problem using these tools. Since we already know about the map of Romania, lets see if we can plan a trip across a simplified map of Romania.
#
# Here is our simplified map definition:

# %%
from aima.utils import *
# this imports the required expr so we can create our knowledge base

knowledge_base = [
    expr("Connected(Bucharest,Pitesti)"),
    expr("Connected(Pitesti,Rimnicu)"),
    expr("Connected(Rimnicu,Sibiu)"),
    expr("Connected(Sibiu,Fagaras)"),
    expr("Connected(Fagaras,Bucharest)"),
    expr("Connected(Pitesti,Craiova)"),
    expr("Connected(Craiova,Rimnicu)")
    ]

# %% [markdown]
# Let us add some logic propositions to complete our knowledge about travelling around the map. These are the typical symmetry and transitivity properties of connections on a map. We can now be sure that our `knowledge_base` understands what it truly means for two locations to be connected in the sense usually meant by humans when we use the term.
#
# Let's also add our starting location - *Sibiu* to the map.

# %%
knowledge_base.extend([
     expr("Connected(x,y) ==> Connected(y,x)"),
     expr("Connected(x,y) & Connected(y,z) ==> Connected(x,z)"),
     expr("At(Sibiu)")
    ])

# %% [markdown]
# We now have a complete knowledge base, which can be seen like this:

# %%
knowledge_base

# %% [markdown]
# We now define possible actions to our problem. We know that we can drive between any connected places. But, as is evident from [this](https://en.wikipedia.org/wiki/List_of_airports_in_Romania) list of Romanian airports, we can also fly directly between Sibiu, Bucharest, and Craiova.
#
# We can define these flight actions like this:

# %%
#Sibiu to Bucharest
precond = 'At(Sibiu)'
effect = 'At(Bucharest) & ~At(Sibiu)'
fly_s_b = Action('Fly(Sibiu, Bucharest)', precond, effect)

#Bucharest to Sibiu
precond = 'At(Bucharest)'
effect = 'At(Sibiu) & ~At(Bucharest)'
fly_b_s = Action('Fly(Bucharest, Sibiu)', precond, effect)

#Sibiu to Craiova
precond = 'At(Sibiu)'
effect = 'At(Craiova) & ~At(Sibiu)'
fly_s_c = Action('Fly(Sibiu, Craiova)', precond, effect)

#Craiova to Sibiu
precond = 'At(Craiova)'
effect = 'At(Sibiu) & ~At(Craiova)'
fly_c_s = Action('Fly(Craiova, Sibiu)', precond, effect)

#Bucharest to Craiova
precond = 'At(Bucharest)'
effect = 'At(Craiova) & ~At(Bucharest)'
fly_b_c = Action('Fly(Bucharest, Craiova)', precond, effect)

#Craiova to Bucharest
precond = 'At(Craiova)'
effect = 'At(Bucharest) & ~At(Craiova)'
fly_c_b = Action('Fly(Craiova, Bucharest)', precond, effect)

# %% [markdown]
# And the drive actions like this.

# %%
#Drive
precond = 'At(x)'
effect = 'At(y) & ~At(x)'
drive = Action('Drive(x, y)', precond, effect)

# %% [markdown]
# Our goal is defined as

# %%
goals = 'At(Bucharest)'


# %% [markdown]
# Finally, we can define a a function that will tell us when we have reached our destination, Bucharest.

# %%
def goal_test(kb):
    return kb.ask(expr('At(Bucharest)'))


# %% [markdown]
# Thus, with all the components in place, we can define the planning problem.

# %%
prob = PlanningProblem(knowledge_base, goals, [fly_s_b, fly_b_s, fly_s_c, fly_c_s, fly_b_c, fly_c_b, drive])

# %% [markdown]
# ## PLANNING PROBLEMS
# ---
#
# ## Air Cargo Problem

# %% [markdown]
# In the Air Cargo problem, we start with cargo at two airports, SFO and JFK. Our goal is to send each cargo to the other airport. We have two airplanes to help us accomplish the task. 
# The problem can be defined with three actions: Load, Unload and Fly. 
# Let us look how the `air_cargo` problem has been defined in the module. 

# %%
psource(air_cargo)

# %% [markdown]
# **At(c, a):** The cargo **'c'** is at airport **'a'**.
#
# **~At(c, a):** The cargo **'c'** is _not_ at airport **'a'**.
#
# **In(c, p):** Cargo **'c'** is in plane **'p'**.
#
# **~In(c, p):** Cargo **'c'** is _not_ in plane **'p'**.
#
# **Cargo(c):** Declare **'c'** as cargo.
#
# **Plane(p):** Declare **'p'** as plane.
#
# **Airport(a):** Declare **'a'** as airport.
#
#
#
# In the `initial_state`, we have cargo C1, plane P1 at airport SFO and cargo C2, plane P2 at airport JFK. 
# Our goal state is to have cargo C1 at airport JFK and cargo C2 at airport SFO. We will discuss on how to achieve this. Let us now define an object of the `air_cargo` problem:

# %%
airCargo = air_cargo()

# %% [markdown]
# Before taking any actions, we will check if `airCargo` has reached its goal:

# %%
print(airCargo.goal_test())

# %% [markdown]
# It returns False because the goal state is not yet reached. Now, we define the sequence of actions that it should take in order to achieve the goal.
# The actions are then carried out on the `airCargo` PlanningProblem.
#
# The actions available to us are the following: Load, Unload, Fly
#
# **Load(c, p, a):** Load cargo **'c'** into plane **'p'** from airport **'a'**.
#
# **Fly(p, f, t):** Fly the plane **'p'** from airport **'f'** to airport **'t'**.
#
# **Unload(c, p, a):** Unload cargo **'c'** from plane **'p'** to airport **'a'**.
#
# This problem can have multiple valid solutions.
# One such solution is shown below.

# %%
solution = [expr("Load(C1 , P1, SFO)"),
            expr("Fly(P1, SFO, JFK)"),
            expr("Unload(C1, P1, JFK)"),
            expr("Load(C2, P2, JFK)"),
            expr("Fly(P2, JFK, SFO)"),
            expr("Unload (C2, P2, SFO)")] 

for action in solution:
    airCargo.act(action)

# %% [markdown]
# As the `airCargo` has taken all the steps it needed in order to achieve the goal, we can now check if it has acheived its goal:

# %%
print(airCargo.goal_test())

# %% [markdown]
# It has now achieved its goal.

# %% [markdown]
# ## The Spare Tire Problem

# %% [markdown]
# Let's consider the problem of changing a flat tire of a car. 
# The goal is to mount a spare tire onto the car's axle, given that we have a flat tire on the axle and a spare tire in the trunk. 

# %%
psource(spare_tire)

# %% [markdown]
# **At(obj, loc):** object **'obj'** is at location **'loc'**.
#
# **~At(obj, loc):** object **'obj'** is _not_ at location **'loc'**.
#
# **Tire(t):** Declare a tire of type **'t'**.
#
# Let us now define an object of `spare_tire` problem:

# %%
spareTire = spare_tire()

# %% [markdown]
# Before taking any actions, we will check if `spare_tire` has reached its goal:

# %%
print(spareTire.goal_test())

# %% [markdown]
# As we can see, it hasn't completed the goal. 
# We now define a possible solution that can help us reach the goal of having a spare tire mounted onto the car's axle. 
# The actions are then carried out on the `spareTire` PlanningProblem.
#
# The actions available to us are the following: Remove, PutOn
#
# **Remove(obj, loc):** Remove the tire **'obj'** from the location **'loc'**.
#
# **PutOn(t, Axle):** Attach the tire **'t'** on the Axle.
#
# **LeaveOvernight():** We live in a particularly bad neighborhood and all tires, flat or not, are stolen if we leave them overnight.
#
#

# %%
solution = [expr("Remove(Flat, Axle)"),
            expr("Remove(Spare, Trunk)"),
            expr("PutOn(Spare, Axle)")]

for action in solution:
    spareTire.act(action)

# %%
print(spareTire.goal_test())

# %% [markdown]
# This is a valid solution.
# <br>
# Another possible solution is

# %%
spareTire = spare_tire()

solution = [expr('Remove(Spare, Trunk)'),
            expr('Remove(Flat, Axle)'),
            expr('PutOn(Spare, Axle)')]

for action in solution:
    spareTire.act(action)

# %%
print(spareTire.goal_test())

# %% [markdown]
# Notice that both solutions work, which means that the problem can be solved irrespective of the order in which the `Remove` actions take place, as long as both `Remove` actions take place before the `PutOn` action.

# %% [markdown]
# We have successfully mounted a spare tire onto the axle.

# %% [markdown]
# ## Three Block Tower Problem

# %% [markdown]
# This problem's domain consists of a set of cube-shaped blocks sitting on a table. 
# The blocks can be stacked, but only one block can fit directly on top of another.
# A robot arm can pick up a block and move it to another position, either on the table or on top of another block. 
# The arm can pick up only one block at a time, so it cannot pick up a block that has another one on it. 
# The goal will always be to build one or more stacks of blocks. 
# In our case, we consider only three blocks.
# The particular configuration we will use is called the Sussman anomaly after Prof. Gerry Sussman.

# %% [markdown]
# Let's take a look at the definition of `three_block_tower()` in the module.

# %%
psource(three_block_tower)

# %% [markdown]
# **On(b, x):** The block **'b'** is on **'x'**. **'x'** can be a table or a block.
#
# **~On(b, x):** The block **'b'** is _not_ on **'x'**. **'x'** can be a table or a block.
#
# **Block(b):** Declares **'b'** as a block.
#
# **Clear(x):** To indicate that there is nothing on **'x'** and it is free to be moved around.
#
# **~Clear(x):** To indicate that there is something on **'x'** and it cannot be moved.
#  
#  Let us now define an object of `three_block_tower` problem:

# %%
threeBlockTower = three_block_tower()

# %% [markdown]
# Before taking any actions, we will check if `threeBlockTower` has reached its goal:

# %%
print(threeBlockTower.goal_test())

# %% [markdown]
# As we can see, it hasn't completed the goal. 
# We now define a sequence of actions that can stack three blocks in the required order. 
# The actions are then carried out on the `threeBlockTower` PlanningProblem.
#
# The actions available to us are the following: MoveToTable, Move
#
# **MoveToTable(b, x): ** Move box **'b'** stacked on **'x'** to the table, given that box **'b'** is clear.
#
# **Move(b, x, y): ** Move box **'b'** stacked on **'x'** to the top of **'y'**, given that both **'b'** and **'y'** are clear.
#

# %%
solution = [expr("MoveToTable(C, A)"),
            expr("Move(B, Table, C)"),
            expr("Move(A, Table, B)")]

for action in solution:
    threeBlockTower.act(action)

# %% [markdown]
# As the `three_block_tower` has taken all the steps it needed in order to achieve the goal, we can now check if it has acheived its goal.

# %%
print(threeBlockTower.goal_test())

# %% [markdown]
# It has now successfully achieved its goal i.e, to build a stack of three blocks in the specified order.

# %% [markdown]
# The `three_block_tower` problem can also be defined in simpler terms using just two actions `ToTable(x, y)` and `FromTable(x, y)`.
# The underlying problem remains the same however, stacking up three blocks in a certain configuration given a particular starting state.
# Let's have a look at the alternative definition.

# %%
psource(simple_blocks_world)

# %% [markdown]
# **On(x, y):** The block **'x'** is on **'y'**. Both **'x'** and **'y'** have to be blocks.
#
# **~On(x, y):** The block **'x'** is _not_ on **'y'**. Both **'x'** and **'y'** have to be blocks.
#
# **OnTable(x):** The block **'x'** is on the table.
#
# **~OnTable(x):** The block **'x'** is _not_ on the table.
#
# **Clear(x):** To indicate that there is nothing on **'x'** and it is free to be moved around.
#
# **~Clear(x):** To indicate that there is something on **'x'** and it cannot be moved.
#
# Let's now define a `simple_blocks_world` prolem.

# %%
simpleBlocksWorld = simple_blocks_world()

# %% [markdown]
# Before taking any actions, we will see if `simple_bw` has reached its goal.

# %%
simpleBlocksWorld.goal_test()

# %% [markdown]
# As we can see, it hasn't completed the goal. 
# We now define a sequence of actions that can stack three blocks in the required order. 
# The actions are then carried out on the `simple_bw` PlanningProblem.
#
# The actions available to us are the following: MoveToTable, Move
#
# **ToTable(x, y): ** Move box **'x'** stacked on **'y'** to the table, given that box **'y'** is clear.
#
# **FromTable(x, y): ** Move box **'x'** from wherever it is, to the top of **'y'**, given that both **'x'** and **'y'** are clear.
#

# %%
solution = [expr('ToTable(A, B)'),
            expr('FromTable(B, A)'),
            expr('FromTable(C, B)')]

for action in solution:
    simpleBlocksWorld.act(action)

# %% [markdown]
# As the `three_block_tower` has taken all the steps it needed in order to achieve the goal, we can now check if it has acheived its goal.

# %%
print(simpleBlocksWorld.goal_test())

# %% [markdown]
# It has now successfully achieved its goal i.e, to build a stack of three blocks in the specified order.

# %% [markdown]
# ## Shopping Problem

# %% [markdown]
# This problem requires us to acquire a carton of milk, a banana and a drill.
# Initially, we start from home and it is known to us that milk and bananas are available in the supermarket and the hardware store sells drills.
# Let's take a look at the definition of the `shopping_problem` in the module.

# %%
psource(shopping_problem)

# %% [markdown]
# **At(x):** Indicates that we are currently at **'x'** where **'x'** can be Home, SM (supermarket) or HW (Hardware store).
#
# **~At(x):** Indicates that we are currently _not_ at **'x'**.
#
# **Sells(s, x):** Indicates that item **'x'** can be bought from store **'s'**.
#
# **Have(x):** Indicates that we possess the item **'x'**.

# %%
shoppingProblem = shopping_problem()

# %% [markdown]
# Let's first check whether the goal state Have(Milk), Have(Banana), Have(Drill) is reached or not.

# %%
print(shoppingProblem.goal_test())

# %% [markdown]
# Let's look at the possible actions
#
# **Buy(x, store):** Buy an item **'x'** from a **'store'** given that the **'store'** sells **'x'**.
#
# **Go(x, y):** Go to destination **'y'** starting from source **'x'**.

# %% [markdown]
# We now define a valid solution that will help us reach the goal.
# The sequence of actions will then be carried out onto the `shoppingProblem` PlanningProblem.

# %%
solution = [expr('Go(Home, SM)'),
            expr('Buy(Milk, SM)'),
            expr('Buy(Banana, SM)'),
            expr('Go(SM, HW)'),
            expr('Buy(Drill, HW)')]

for action in solution:
    shoppingProblem.act(action)

# %% [markdown]
# We have taken the steps required to acquire all the stuff we need. 
# Let's see if we have reached our goal.

# %%
shoppingProblem.goal_test()

# %% [markdown]
# It has now successfully achieved the goal.

# %% [markdown]
# ## Socks and Shoes

# %% [markdown]
# This is a simple problem of putting on a pair of socks and shoes.
# The problem is defined in the module as given below.

# %%
psource(socks_and_shoes)

# %% [markdown]
# **LeftSockOn:** Indicates that we have already put on the left sock.
#
# **RightSockOn:** Indicates that we have already put on the right sock.
#
# **LeftShoeOn:** Indicates that we have already put on the left shoe.
#
# **RightShoeOn:** Indicates that we have already put on the right shoe.
#

# %%
socksShoes = socks_and_shoes()

# %% [markdown]
# Let's first check whether the goal state is reached or not.

# %%
socksShoes.goal_test()

# %% [markdown]
# As the goal state isn't reached, we will define a sequence of actions that might help us achieve the goal.
# These actions will then be acted upon the `socksShoes` PlanningProblem to check if the goal state is reached.

# %%
solution = [expr('RightSock'),
            expr('RightShoe'),
            expr('LeftSock'),
            expr('LeftShoe')]

# %%
for action in solution:
    socksShoes.act(action)
    
socksShoes.goal_test()

# %% [markdown]
# We have reached our goal.

# %% [markdown]
# ## Cake Problem

# %% [markdown]
# This problem requires us to reach the state of having a cake and having eaten a cake simlutaneously, given a single cake.
# Let's first take a look at the definition of the `have_cake_and_eat_cake_too` problem in the module.

# %%
psource(have_cake_and_eat_cake_too)

# %% [markdown]
# Since this problem doesn't involve variables, states can be considered similar to symbols in propositional logic.
#
# **Have(Cake):** Declares that we have a **'Cake'**.
#
# **~Have(Cake):** Declares that we _don't_ have a **'Cake'**.

# %%
cakeProblem = have_cake_and_eat_cake_too()

# %% [markdown]
# First let us check whether the goal state 'Have(Cake)' and 'Eaten(Cake)' are reached or not.

# %%
print(cakeProblem.goal_test())

# %% [markdown]
# Let us look at the possible actions.
#
# **Bake(x):** To bake **' x '**.
#
# **Eat(x):** To eat **' x '**.

# %% [markdown]
# We now define a valid solution that can help us reach the goal.
# The sequence of actions will then be acted upon the `cakeProblem` PlanningProblem.

# %%
solution = [expr("Eat(Cake)"),
            expr("Bake(Cake)")]

for action in solution:
    cakeProblem.act(action)

# %% [markdown]
# Now we have made actions to bake the cake and eat the cake. Let us check if we have reached the goal.

# %%
print(cakeProblem.goal_test())

# %% [markdown]
# It has now successfully achieved its goal i.e, to have and eat the cake.

# %% [markdown]
# One might wonder if the order of the actions matters for this problem.
# Let's see for ourselves.

# %%
cakeProblem = have_cake_and_eat_cake_too()

solution = [expr('Eat(Cake)'),
            expr('Bake(Cake)')]

for action in solution:
    cakeProblem.act(action)

# %% [markdown]
# It raises an exception.
# Indeed, according to the problem, we cannot bake a cake if we already have one.
# In planning terms, '~Have(Cake)' is a precondition to the action 'Bake(Cake)'.
# Hence, this solution is invalid.

# %% [markdown]
# ## PLANNING IN THE REAL WORLD
# ---
# ## PROBLEM
# The `Problem` class is a wrapper for `PlanningProblem` with some additional functionality and data-structures to handle real-world planning problems that involve time and resource constraints.
# The `Problem` class includes everything that the `PlanningProblem` class includes.
# Additionally, it also includes the following attributes essential to define a real-world planning problem:
# - a list of `jobs` to be done
# - a dictionary of `resources`
#
# It also overloads the `act` method to call the `do_action` method of the `HLA` class, 
# and also includes a new method `refinements` that finds refinements or primitive actions for high level actions.
# <br>
# `hierarchical_search` and `angelic_search` are also built into the `Problem` class to solve such planning problems.

# %%
psource(PlanningProblem)

# %% [markdown]
# ## HLA
# To be able to model a real-world planning problem properly, it is  essential to be able to represent a _high-level action (HLA)_ that can be hierarchically reduced to primitive actions.

# %%
psource(HLA)

# %% [markdown]
# In addition to preconditions and effects, an object of the `HLA` class also stores:
# - the `duration` of the HLA
# - the quantity of consumption of _consumable_ resources
# - the quantity of _reusable_ resources used
# - a bool `completed` denoting if the `HLA` has been completed
#
# The class also has some useful helper methods:
# - `do_action`: checks if required consumable and reusable resources are available and if so, executes the action.
# - `has_consumable_resource`: checks if there exists sufficient quantity of the required consumable resource.
# - `has_usable_resource`: checks if reusable resources are available and not already engaged.
# - `inorder`: ensures that all the jobs that had to be executed before the current one have been successfully executed.

# %% [markdown]
# ## PLANNING PROBLEMS
# ---
# ## Job-shop Problem
# This is a simple problem involving the assembly of two cars simultaneously.
# The problem consists of two jobs, each of the form [`AddEngine`, `AddWheels`, `Inspect`] to be performed on two cars with different requirements and availability of resources.
# <br>
# Let's look at how the `job_shop_problem` has been defined on the  module.

# %%
psource(job_shop_problem)

# %% [markdown]
# The states of this problem are:
# <br>
# <br>
# **Has(x, y)**: Car **'x'** _has_ **'y'** where **'y'** can be an Engine or a Wheel.
#
# **~Has(x, y)**: Car **'x'** does _not have_ **'y'** where **'y'** can be an Engine or a Wheel.
#
# **Inspected(c)**: Car **'c'** has been _inspected_.
#
# **~Inspected(c)**: Car **'c'** has _not_ been inspected.
#
# In the initial state, `C1` and `C2` are cars and neither have an engine or wheels and haven't been inspected.
# `E1` and `E2` are engines.
# `W1` and `W2` are wheels.
# <br>
# Our goal is to have engines and wheels on both cars and to get them inspected. We will discuss how to achieve this.
# <br>
# Let's define an object of the `job_shop_problem`.

# %%
jobShopProblem = job_shop_problem()

# %% [markdown]
# Before taking any actions, we will check if `jobShopProblem` has reached its goal.

# %%
print(jobShopProblem.goal_test())

# %% [markdown]
# We now define a possible solution that can help us reach the goal. 
# The actions are then carried out on the `jobShopProblem` object.

# %% [markdown]
# The following actions are available to us:
#
# **AddEngine1**: Adds an engine to the car C1. Takes 30 minutes to complete and uses an engine hoist.
#  
# **AddEngine2**: Adds an engine to the car C2. Takes 60 minutes to complete and uses an engine hoist.
#
# **AddWheels1**: Adds wheels to car C1. Takes 30 minutes to complete. Uses a wheel station and consumes 20 lug nuts.
#
# **AddWheels2**: Adds wheels to car C2. Takes 15 minutes to complete. Uses a wheel station and consumes 20 lug nuts as well.
#
# **Inspect1**: Gets car C1 inspected. Requires 10 minutes of inspection by one inspector.
#
# **Inspect2**: Gets car C2 inspected. Requires 10 minutes of inspection by one inspector.

# %%
solution = [jobShopProblem.jobs[1][0],
            jobShopProblem.jobs[1][1],
            jobShopProblem.jobs[1][2],
            jobShopProblem.jobs[0][0],
            jobShopProblem.jobs[0][1],
            jobShopProblem.jobs[0][2]]

for action in solution:
    jobShopProblem.act(action)

# %%
print(jobShopProblem.goal_test())

# %% [markdown]
# This is a valid solution and one of many correct ways to solve this problem.

# %% [markdown]
# ## Double tennis problem
# This problem is a simple case of a multiactor planning problem, where two agents act at once and can simultaneously change the current state of the problem. 
# A correct plan is one that, if executed by the actors, achieves the goal.
# In the true multiagent setting, of course, the agents may not agree to execute any particular plan, but atleast they will know what plans _would_ work if they _did_ agree to execute them.
# <br>
# In the double tennis problem, two actors A and B are playing together and can be in one of four locations: `LeftBaseLine`, `RightBaseLine`, `LeftNet` and `RightNet`.
# The ball can be returned only if a player is in the right place.
# Each action must include the actor as an argument.
# <br>
# Let's first look at the definition of the `double_tennis_problem` in the module.

# %%
psource(double_tennis_problem)

# %% [markdown]
# The states of this problem are:
#
# **Approaching(Ball, loc)**: The `Ball` is approaching the location `loc`.
#
# **Returned(Ball)**: One of the actors successfully hit the approaching ball from the correct location which caused it to return to the other side.
#
# **At(actor, loc)**: `actor` is at location `loc`.
#
# **~At(actor, loc)**: `actor` is _not_ at location `loc`.
#
# Let's now define an object of `double_tennis_problem`.
#

# %%
doubleTennisProblem = double_tennis_problem()

# %% [markdown]
# Before taking any actions, we will check if `doubleTennisProblem` has reached the goal.

# %%
print(doubleTennisProblem.goal_test())

# %% [markdown]
# As we can see, the goal hasn't been reached. 
# We now define a possible solution that can help us reach the goal of having the ball returned.
# The actions will then be carried out on the `doubleTennisProblem` object.

# %% [markdown]
# The actions available to us are the following:
#
# **Hit(actor, ball, loc)**: returns an approaching ball if `actor` is present at the `loc` that the ball is approaching.
#
# **Go(actor, to, loc)**: moves an `actor` from location `loc` to location `to`.
#
# We notice something different in this problem though, 
# which is quite unlike any other problem we have seen so far. 
# The goal state of the problem contains a variable `a`.
# This happens sometimes in multiagent planning problems 
# and it means that it doesn't matter _which_ actor is at the `LeftNet` or the `RightNet`, as long as there is atleast one actor at either `LeftNet` or `RightNet`.

# %%
solution = [expr('Go(A, RightBaseLine, LeftBaseLine)'),
            expr('Hit(A, Ball, RightBaseLine)'),
            expr('Go(A, LeftNet, RightBaseLine)')]

for action in solution:
    doubleTennisProblem.act(action)

# %%
doubleTennisProblem.goal_test()

# %% [markdown]
# It has now successfully reached its goal, ie, to return the approaching ball.
