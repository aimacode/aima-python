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
# # Hierarchical Search 
#
# Hierarchical search is a a planning algorithm in high level of abstraction. <br>
# Instead of actions as in classical planning (chapter 10) (primitive actions) we now use high level actions (HLAs) (see planning.ipynb) <br>
#
# ## Refinements
#
# Each __HLA__ has one or more refinements into a sequence of actions, each of which may be an HLA or a primitive action (which has no refinements by definition).<br>
# For example:
# -  (a) the high level action "Go to San Fransisco airport" (Go(Home, SFO)), might have two possible refinements, "Drive to San Fransisco airport" and "Taxi to San Fransisco airport". 
# <br>
# -  (b) A recursive refinement for navigation in the vacuum world would be: to get to a
# destination, take a step, and then go to the destination.
# <br>
# ![title](images/refinement.png)
# <br>
# -  __implementation__: An HLA refinement that contains only primitive actions is called an implementation of the HLA
# -  An implementation of a high-level plan (a sequence of HLAs) is the concatenation of implementations of each HLA in the sequence
# - A high-level plan __achieves the goal__ from a given state if at least one of its implementations achieves the goal from that state
# <br>
#
# The refinements function input is: 
# -  __hla__: the HLA of which we want to compute its refinements
# - __state__: the knoweledge base of the current problem (Problem.init)
# - __library__: the hierarchy of the actions in the planning problem
#
#

# %%
from aima.planning import * 
from aima.notebook_utils import psource

# %%
psource(RealWorldPlanningProblem.refinements)

# %% [markdown]
# ## Hierarchical search 
#
# Hierarchical search is a breadth-first implementation of hierarchical forward planning search in the space of refinements. (i.e. repeatedly choose an HLA in the current plan and replace it with one of its refinements, until the plan achieves the goal.) 
#
# <br>
# The algorithms input is: problem and hierarchy
# -  __problem__: is of type Problem 
# -  __hierarchy__: is a dictionary consisting of all the actions and the order in which they are performed. 
# <br>
#
# In top level call, initial_plan contains [act] (i.e. is the action to be performed)   

# %%
psource(RealWorldPlanningProblem.hierarchical_search)

# %% [markdown]
# ## Example
#
# Suppose that somebody wants to get to the airport. 
# The possible ways to do so is either get a taxi, or drive to the airport. <br>
# Those two actions have some preconditions and some effects. 
# If you get the taxi, you need to have cash, whereas if you drive you need to have a car. <br>
# Thus we define the following hierarchy of possible actions.
#
# ##### hierarchy

# %%
library = {
        'HLA': ['Go(Home,SFO)', 'Go(Home,SFO)', 'Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)', 'Taxi(Home, SFO)'],
        'steps': [['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'], ['Taxi(Home, SFO)'], [], [], []],
        'precond': [['At(Home) & Have(Car)'], ['At(Home)'], ['At(Home) & Have(Car)'], ['At(SFOLongTermParking)'], ['At(Home)']],
        'effect': [['At(SFO) & ~At(Home)'], ['At(SFO) & ~At(Home) & ~Have(Cash)'], ['At(SFOLongTermParking) & ~At(Home)'], ['At(SFO) & ~At(LongTermParking)'], ['At(SFO) & ~At(Home) & ~Have(Cash)']] }



# %% [markdown]
#
# the possible actions are the following:

# %%
go_SFO = HLA('Go(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
taxi_SFO = HLA('Taxi(Home,SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home) & ~Have(Cash)')
drive_SFOLongTermParking = HLA('Drive(Home, SFOLongTermParking)', 'At(Home) & Have(Car)','At(SFOLongTermParking) & ~At(Home)' )
shuttle_SFO = HLA('Shuttle(SFOLongTermParking, SFO)', 'At(SFOLongTermParking)', 'At(SFO) & ~At(LongTermParking)')

# %% [markdown]
# Suppose that (our preconditionds are that) we are Home and we have cash and car and  our goal is to get to SFO and maintain our cash, and our possible actions are the above. <br>
# ##### Then our problem is: 

# %%
prob = RealWorldPlanningProblem('At(Home) & Have(Cash) & Have(Car)', 'At(SFO) & Have(Cash)', [go_SFO])

# %% [markdown]
# ##### Refinements
#
# The refinements of the action Go(Home, SFO), are defined as: <br>
# ['Drive(Home,SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'], ['Taxi(Home, SFO)']

# %%
for sequence in RealWorldPlanningProblem.refinements(go_SFO, library):
    print (sequence)
    print([x.__dict__ for x in sequence ], '\n')

# %% [markdown]
# Run the hierarchical search
# ##### Top level call

# %%
plan= RealWorldPlanningProblem.hierarchical_search(prob, library)
print (plan, '\n')
print ([x.__dict__ for x in plan])

# %% [markdown]
# ## Example 2

# %%
library_2 = {
        'HLA': ['Go(Home,SFO)', 'Go(Home,SFO)', 'Bus(Home, MetroStop)', 'Metro(MetroStop, SFO)' , 'Metro(MetroStop, SFO)', 'Metro1(MetroStop, SFO)', 'Metro2(MetroStop, SFO)'  ,'Taxi(Home, SFO)'],
        'steps': [['Bus(Home, MetroStop)', 'Metro(MetroStop, SFO)'], ['Taxi(Home, SFO)'], [], ['Metro1(MetroStop, SFO)'], ['Metro2(MetroStop, SFO)'],[],[],[]],
        'precond': [['At(Home)'], ['At(Home)'], ['At(Home)'], ['At(MetroStop)'], ['At(MetroStop)'],['At(MetroStop)'], ['At(MetroStop)'] ,['At(Home) & Have(Cash)']],
        'effect': [['At(SFO) & ~At(Home)'], ['At(SFO) & ~At(Home) & ~Have(Cash)'], ['At(MetroStop) & ~At(Home)'], ['At(SFO) & ~At(MetroStop)'], ['At(SFO) & ~At(MetroStop)'], ['At(SFO) & ~At(MetroStop)'] , ['At(SFO) & ~At(MetroStop)'] ,['At(SFO) & ~At(Home) & ~Have(Cash)']] 
        }

# %%
plan_2 = RealWorldPlanningProblem.hierarchical_search(prob, library_2)
print(plan_2, '\n')
print([x.__dict__ for x in plan_2])
