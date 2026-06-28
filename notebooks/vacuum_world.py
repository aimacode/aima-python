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
# # THE VACUUM WORLD   
#
# In this notebook, we will be discussing **the structure of agents** through an example of the **vacuum agent**. The job of AI is to design an **agent program** that implements the agent function: the mapping from percepts to actions. We assume this program will run on some sort of computing device with physical sensors and actuators: we call this the **architecture**:
#
# <h3 align="center">agent = architecture + program</h3>

# %% [markdown]
# Before moving on, please review [<b>agents.ipynb</b>](https://github.com/aimacode/aima-python/blob/master/agents.ipynb)

# %% [markdown]
# ## CONTENTS
#
# * Agent
# * Random Agent Program
# * Table-Driven Agent Program
# * Simple Reflex Agent Program
# * Model-Based Reflex Agent Program
# * Goal-Based Agent Program
# * Utility-Based Agent Program
# * Learning Agent

# %% [markdown]
# ## AGENT PROGRAMS
#
# An agent program takes the current percept as input from the sensors and returns an action to the actuators. There is a difference between an agent program and an agent function: an agent program takes the current percept as input whereas an agent function takes the entire percept history.
#
# The agent program takes just the current percept as input because nothing more is available from the environment; if the agent's actions depend on the entire percept sequence, the agent will have to remember the percept.
#
# We'll discuss the following agent programs here with the help of the vacuum world example:
#
# * Random Agent Program
# * Table-Driven Agent Program
# * Simple Reflex Agent Program
# * Model-Based Reflex Agent Program
# * Goal-Based Agent Program
# * Utility-Based Agent Program

# %% [markdown]
# ## Random Agent Program
#
# A random agent program, as the name suggests, chooses an action at random, without taking into account the percepts.   
# Here, we will demonstrate a random vacuum agent for a trivial vacuum environment, that is, the two-state environment.

# %% [markdown]
# Let's begin by importing all the functions from the agents module:

# %%
from aima.agents import *
from aima.notebook_utils import psource

# %% [markdown]
# Let us first see how we define the TrivialVacuumEnvironment. Run the next cell to see how abstract class TrivialVacuumEnvironment is defined in agents module:

# %%
psource(TrivialVacuumEnvironment)

# %%
# These are the two locations for the two-state environment
loc_A, loc_B = (0, 0), (1, 0)

# Initialize the two-state environment
trivial_vacuum_env = TrivialVacuumEnvironment()

# Check the initial state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))

# %% [markdown]
# Let's create our agent now. This agent will choose any of the actions from 'Right', 'Left', 'Suck' and 'NoOp' (No Operation) randomly.

# %%
# Create the random agent
random_agent = Agent(program=RandomAgentProgram(['Right', 'Left', 'Suck', 'NoOp']))

# %% [markdown]
# We will now add our agent to the environment.

# %%
# Add agent to the environment
trivial_vacuum_env.add_thing(random_agent)

print("RandomVacuumAgent is located at {}.".format(random_agent.location))

# %% [markdown]
# Let's run our environment now.

# %%
# Running the environment
trivial_vacuum_env.step()

# Check the current state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))

print("RandomVacuumAgent is located at {}.".format(random_agent.location))

# %% [markdown]
# ## TABLE-DRIVEN AGENT PROGRAM
#
# A table-driven agent program keeps track of the percept sequence and then uses it to index into a table of actions to decide what to do. The table represents explicitly the agent function that the agent program embodies.  
# In the two-state vacuum world, the table would consist of all the possible states of the agent.

# %%
table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Dirty'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean')): 'Left',
             ((loc_A, 'Dirty'), (loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'
        }

# %% [markdown]
# We will now create a table-driven agent program for our two-state environment.

# %%
# Create a table-driven agent
table_driven_agent = Agent(program=TableDrivenAgentProgram(table=table))

# %% [markdown]
# Since we are using the same environment, let's remove the previously added random agent from the environment to avoid confusion.

# %%
trivial_vacuum_env.delete_thing(random_agent)

# %%
# Add the table-driven agent to the environment
trivial_vacuum_env.add_thing(table_driven_agent)

print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))

# %%
# Run the environment
trivial_vacuum_env.step()

# Check the current state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))

print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))

# %% [markdown]
# ## SIMPLE REFLEX AGENT PROGRAM
#
# A simple reflex agent program selects actions on the basis of the *current* percept, ignoring the rest of the percept history. These agents work on a **condition-action rule** (also called **situation-action rule**, **production** or **if-then rule**), which tells the agent the action to trigger when a particular situation is encountered.  
#
# The schematic diagram shown in **Figure 2.9** of the book will make this more clear:
#
# "![simple reflex agent](images/simple_reflex_agent.jpg)"

# %% [markdown]
# Let us now create a simple reflex agent for the environment.

# %%
# Delete the previously added table-driven agent
trivial_vacuum_env.delete_thing(table_driven_agent)

# %% [markdown]
# To create our agent, we need two functions: INTERPRET-INPUT function, which generates an abstracted description of the current state from the percerpt and the RULE-MATCH function, which returns the first rule in the set of rules that matches the given state description.

# %%

loc_A = (0, 0)
loc_B = (1, 0)

"""We change the simpleReflexAgentProgram so that it doesn't make use of the Rule class"""
def SimpleReflexAgentProgram():
    """This agent takes action based solely on the percept. [Figure 2.10]"""
    
    def program(percept):
        loc, status = percept
        return ('Suck' if status == 'Dirty' 
                else'Right' if loc == loc_A 
                            else'Left')
    return program

        
# Create a simple reflex agent the two-state environment
program = SimpleReflexAgentProgram()
simple_reflex_agent = Agent(program)

# %% [markdown]
# Now add the agent to the environment:

# %%
trivial_vacuum_env.add_thing(simple_reflex_agent)

print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))

# %%
# Run the environment
trivial_vacuum_env.step()

# Check the current state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))

print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))

# %% [markdown]
# ## MODEL-BASED REFLEX AGENT PROGRAM
#
# A model-based reflex agent maintains some sort of **internal state** that depends on the percept history and thereby reflects at least some of the unobserved aspects of the current state. In addition to this, it also requires a **model** of the world, that is, knowledge about "how the world works".
#
# The schematic diagram shown in **Figure 2.11** of the book will make this more clear:
# <img src="images/model_based_reflex_agent.jpg">

# %% [markdown]
# We will now create a model-based reflex agent for the environment:

# %%
# Delete the previously added simple reflex agent
trivial_vacuum_env.delete_thing(simple_reflex_agent)


# %% [markdown]
# We need another function UPDATE-STATE which will be responsible for creating a new state description.

# %%
# TODO: Implement this function for the two-dimensional environment
def update_state(state, action, percept, model):
    pass

# Create a model-based reflex agent
model_based_reflex_agent = ModelBasedVacuumAgent()

# Add the agent to the environment
trivial_vacuum_env.add_thing(model_based_reflex_agent)

print("ModelBasedVacuumAgent is located at {}.".format(model_based_reflex_agent.location))

# %%
# Run the environment
trivial_vacuum_env.step()

# Check the current state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))

print("ModelBasedVacuumAgent is located at {}.".format(model_based_reflex_agent.location))

# %% [markdown]
# ## GOAL-BASED AGENT PROGRAM
#
# A goal-based agent needs some sort of **goal** information that describes situations that are desirable, apart from the current state description.
#
# **Figure 2.13** of the book shows a model-based, goal-based agent:
# <img src="images/model_goal_based_agent.jpg">
#
# **Search** (Chapters 3 to 5) and **Planning** (Chapters 10 to 11) are the subfields of AI devoted to finding action sequences that achieve the agent's goals.
#
# ## UTILITY-BASED AGENT PROGRAM
#
# A utility-based agent maximizes its **utility** using the agent's **utility function**, which is essentially an internalization of the agent's performance measure.
#
# **Figure 2.14** of the book shows a model-based, utility-based agent:
# <img src="images/model_utility_based_agent.jpg">

# %% [markdown]
# ## LEARNING AGENT
#
# Learning allows the agent to operate in initially unknown environments and to become more competent than its initial knowledge alone might allow. Here, we will breifly introduce the main ideas of learning agents.  
#
# A learning agent can be divided into four conceptual components. The **learning element** is responsible for making improvements. It uses the feedback from the **critic** on how the agent is doing and determines how the performance element should be modified to do better in the future. The **performance element** is responsible for selecting external actions for the agent: it takes in percepts and decides on actions. The critic tells the learning element how well the agent is doing with respect to a fixed performance standard. It is necesaary because the percepts themselves provide no indication of the agent's success. The last component of the learning agent is the **problem generator**. It is responsible for suggesting actions that will lead to new and informative experiences.  
#
# **Figure 2.15** of the book sums up the components and their working:  
# <img src="images/general_learning_agent.jpg">
