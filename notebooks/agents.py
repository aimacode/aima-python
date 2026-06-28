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
# # Intelligent Agents #
#
# This notebook serves as supporting material for topics covered in **Chapter 2 - Intelligent Agents** from the book *Artificial Intelligence: A Modern Approach.* This notebook uses implementations from [agents.py](https://github.com/aimacode/aima-python/blob/master/agents.py) module. Let's start by importing everything from agents module.

# %%
from aima.agents import *
from aima.notebook_utils import psource

# %% [markdown]
# ## CONTENTS
#
# * Overview
# * Agent
# * Environment
# * Simple Agent and Environment
# * Agents in a 2-D Environment
# * Wumpus Environment
#
# ## OVERVIEW
#
# An agent, as defined in 2.1, is anything that can perceive its <b>environment</b> through sensors, and act upon that environment through actuators based on its <b>agent program</b>. This can be a dog, a robot, or even you. As long as you can perceive the environment and act on it, you are an agent. This notebook will explain how to implement a simple agent, create an environment, and implement a program that helps the agent act on the environment based on its percepts.
#
# ## AGENT
#
# Let us now see how we define an agent. Run the next cell to see how `Agent` is defined in agents module.

# %%
psource(Agent)

# %% [markdown]
# The `Agent` has two methods.
# * `__init__(self, program=None)`: The constructor defines various attributes of the Agent. These include
#
#     * `alive`: which keeps track of whether the agent is alive or not 
#     
#     * `bump`: which tracks if the agent collides with an edge of the environment (for eg, a wall in a park)
#     
#     * `holding`: which is a list containing the `Things` an agent is holding, 
#     
#     * `performance`: which evaluates the performance metrics of the agent 
#     
#     * `program`: which is the agent program and maps an agent's percepts to actions in the environment. If no implementation is provided, it defaults to asking the user to provide actions for each percept.
#     
# * `can_grab(self, thing)`: Is used when an environment contains things that an agent can grab and carry. By default, an agent can carry nothing.
#
# ## ENVIRONMENT
# Now, let us see how environments are defined. Running the next cell will display an implementation of the abstract `Environment` class.

# %%
psource(Environment)


# %% [markdown]
# `Environment` class has lot of methods! But most of them are incredibly simple, so let's see the ones we'll be using in this notebook.
#
# * `thing_classes(self)`: Returns a static array of `Thing` sub-classes that determine what things are allowed in the environment and what aren't
#
# * `add_thing(self, thing, location=None)`: Adds a thing to the environment at location
#
# * `run(self, steps)`: Runs an environment with the agent in it for a given number of steps.
#
# * `is_done(self)`: Returns true if the objective of the agent and the environment has been completed
#
# The next two functions must be implemented by each subclasses of `Environment` for the agent to receive percepts and execute actions 
#
# * `percept(self, agent)`: Given an agent, this method returns a list of percepts that the agent sees at the current time
#
# * `execute_action(self, agent, action)`: The environment reacts to an action performed by a given agent. The changes may result in agent experiencing new percepts or other elements reacting to agent input.

# %% [markdown]
# ## SIMPLE AGENT AND ENVIRONMENT
#
# Let's begin by using the `Agent` class to creating our first agent - a blind dog.

# %%
class BlindDog(Agent):
    def eat(self, thing):
        print("Dog: Ate food at {}.".format(self.location))
            
    def drink(self, thing):
        print("Dog: Drank water at {}.".format( self.location))

dog = BlindDog()

# %% [markdown]
# What we have just done is create a dog who can only feel what's in his location (since he's blind), and can eat or drink. Let's see if he's alive...

# %%
print(dog.alive)


# %% [markdown]
# ![Cool dog](https://gifgun.files.wordpress.com/2015/07/wpid-wp-1435860392895.gif)
# This is our dog. How cool is he? Well, he's hungry and needs to go search for food. For him to do this, we need to give him a program. But before that, let's create a park for our dog to play in.

# %% [markdown]
# ### ENVIRONMENT - Park
#
# A park is an example of an environment because our dog can perceive and act upon it. The <b>Environment</b> class is an abstract class, so we will have to create our own subclass from it before we can use it.

# %%
class Food(Thing):
    pass

class Water(Thing):
    pass

class Park(Environment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == "move down":
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.movedown()
        elif action == "eat":
            items = self.list_things_at(agent.location, tclass=Food)
            if len(items) != 0:
                if agent.eat(items[0]): #Have the dog eat the first item
                    print('{} ate {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the Park after.
        elif action == "drink":
            items = self.list_things_at(agent.location, tclass=Water)
            if len(items) != 0:
                if agent.drink(items[0]): #Have the dog drink the first item
                    print('{} drank {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the Park after.

    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but to prevent killing our cute dog, we will stop before itself - when there is no more food or water'''
        no_edibles = not any(isinstance(thing, Food) or isinstance(thing, Water) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles



# %% [markdown]
# ### PROGRAM - BlindDog
# Now that we have a <b>Park</b> Class, we re-implement our <b>BlindDog</b> to be able to move down and eat food or drink water only if it is present.
#

# %%
class BlindDog(Agent):
    location = 1
    
    def movedown(self):
        self.location += 1
        
    def eat(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Food):
            return True
        return False
    
    def drink(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Water):
            return True
        return False


# %% [markdown]
# Now its time to implement a <b>program</b> module for our dog. A program controls how the dog acts upon its environment. Our program will be very simple, and is shown in the table below.
# <table>
#     <tr>
#         <td><b>Percept:</b> </td>
#         <td>Feel Food </td>
#         <td>Feel Water</td>
#         <td>Feel Nothing</td>
#    </tr>
#    <tr>
#        <td><b>Action:</b> </td>
#        <td>eat</td>
#        <td>drink</td>
#        <td>move down</td>
#    </tr>
#         
# </table>

# %%
def program(percepts):
    '''Returns an action based on the dog's percepts'''
    for p in percepts:
        if isinstance(p, Food):
            return 'eat'
        elif isinstance(p, Water):
            return 'drink'
    return 'move down'


# %% [markdown]
# Let's now run our simulation by creating a park with some food, water, and our dog.

# %%
park = Park()
dog = BlindDog(program)
dogfood = Food()
water = Water()
park.add_thing(dog, 1)
park.add_thing(dogfood, 5)
park.add_thing(water, 7)

park.run(5)

# %% [markdown]
# Notice that the dog moved from location 1 to 4, over 4 steps, and ate food at location 5 in the 5th step.
#
# Let's continue this simulation for 5 more steps.

# %%
park.run(5)

# %% [markdown]
# Perfect! Note how the simulation stopped after the dog drank the water - exhausting all the food and water ends our simulation, as we had defined before. Let's add some more water and see if our dog can reach it.

# %%
park.add_thing(water, 15)
park.run(10)


# %% [markdown]
# Above, we learnt to implement an agent, its program, and an environment on which it acts. However, this was a very simple case. Let's try to add complexity to it by creating a 2-Dimensional environment!
#
#
# ## AGENTS IN A 2D ENVIRONMENT
#
# For us to not read so many logs of what our dog did, we add a bit of graphics while making our Park 2D. To do so, we will need to make it a subclass of <b>GraphicEnvironment</b> instead of Environment. Parks implemented by subclassing <b>GraphicEnvironment</b> class adds these extra properties to it:
#
#  - Our park is indexed in the 4th quadrant of the X-Y plane.
#  - Every time we create a park subclassing <b>GraphicEnvironment</b>, we need to define the colors of all the things we plan to put into the park. The colors are defined in typical [<b>RGB digital 8-bit format</b>](https://en.wikipedia.org/wiki/RGB_color_model#Numeric_representations), common across the web.
#  - Fences are added automatically to all parks so that our dog does not go outside the park's boundary - it just isn't safe for blind dogs to be outside the park by themselves! <b>GraphicEnvironment</b> provides `is_inbounds` function to check if our dog tries to leave the park.
#  
# First let us try to upgrade our 1-dimensional `Park` environment by just replacing its superclass by `GraphicEnvironment`. 

# %%
class Park2D(GraphicEnvironment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == "move down":
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.movedown()
        elif action == "eat":
            items = self.list_things_at(agent.location, tclass=Food)
            if len(items) != 0:
                if agent.eat(items[0]): #Have the dog eat the first item
                    print('{} ate {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the Park after.
        elif action == "drink":
            items = self.list_things_at(agent.location, tclass=Water)
            if len(items) != 0:
                if agent.drink(items[0]): #Have the dog drink the first item
                    print('{} drank {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the Park after.
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but to prevent killing our cute dog, we will stop before itself - when there is no more food or water'''
        no_edibles = not any(isinstance(thing, Food) or isinstance(thing, Water) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles

class BlindDog(Agent):
    location = [0,1] # change location to a 2d value
    direction = Direction("down") # variable to store the direction our dog is facing
    
    def movedown(self):
        self.location[1] += 1
        
    def eat(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Food):
            return True
        return False
    
    def drink(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Water):
            return True
        return False


# %% [markdown]
# Now let's test this new park with our same dog, food and water. We color our dog with a nice red and mark food and water with orange and blue respectively.

# %%
park = Park2D(5,20, color={'BlindDog': (200,0,0), 'Water': (0, 200, 200), 'Food': (230, 115, 40)}) # park width is set to 5, and height to 20
dog = BlindDog(program)
dogfood = Food()
water = Water()
park.add_thing(dog, [0,1])
park.add_thing(dogfood, [0,5])
park.add_thing(water, [0,7])
morewater = Water()
park.add_thing(morewater, [0,15])
print("BlindDog starts at (1,1) facing downwards, lets see if he can find any food!")
park.run(20)

# %% [markdown]
# Adding some graphics was a good idea! We immediately see that the code works, but our blind dog doesn't make any use of the 2 dimensional space available to him. Let's make our dog more energetic so that he turns and moves forward, instead of always moving down. In doing so, we'll also need to make some changes to our environment to be able to handle this extra motion.
#
# ### PROGRAM - EnergeticBlindDog
#
# Let's make our dog turn or move forwards at random - except when he's at the edge of our park - in which case we make him change his direction explicitly by turning to avoid trying to leave the park. However, our dog is blind so he wouldn't know which way to turn - he'd just have to try arbitrarily.
#
# <table>
#     <tr>
#         <td><b>Percept:</b> </td>
#         <td>Feel Food </td>
#         <td>Feel Water</td>
#         <td>Feel Nothing</td>
#    </tr>
#    <tr>
#        <td><b>Action:</b> </td>
#        <td>eat</td>
#        <td>drink</td>
#        <td>
#        <table>
#            <tr>
#                <td><b>Remember being at Edge : </b></td>
#                <td>At Edge</td>
#                <td>Not at Edge</td>
#            </tr>
#            <tr>
#                <td><b>Action : </b></td>
#                <td>Turn Left / Turn Right <br> ( 50% - 50% chance )</td>
#                <td>Turn Left / Turn Right / Move Forward <br> ( 25% - 25% - 50% chance )</td>
#            </tr>
#        </table>
#        </td>
#    </tr>
#         
# </table>

# %%
from random import choice

class EnergeticBlindDog(Agent):
    location = [0,1]
    direction = Direction("down")
    
    def moveforward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
    
    def turn(self, d):
        self.direction = self.direction + d
        
    def eat(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Food):
            return True
        return False
    
    def drink(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Water):
            return True
        return False
        
def program(percepts):
    '''Returns an action based on it's percepts'''
        
    for p in percepts: # first eat or drink - you're a dog!
        if isinstance(p, Food):
            return 'eat'
        elif isinstance(p, Water):
            return 'drink'
        if isinstance(p,Bump): # then check if you are at an edge and have to turn
            turn = False
            choice = random.choice((1,2));
        else:
            choice = random.choice((1,2,3,4)) # 1-right, 2-left, others-forward
    if choice == 1:
        return 'turnright'
    elif choice == 2:
        return 'turnleft'
    else:
        return 'moveforward'
    


# %% [markdown]
# ### ENVIRONMENT - Park2D
#
# We also need to modify our park accordingly, in order to be able to handle all the new actions our dog wishes to execute. Additionally, we'll need to prevent our dog from moving to locations beyond our park boundary - it just isn't safe for blind dogs to be outside the park by themselves.

# %%
class Park2D(GraphicEnvironment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(agent.location) # find out the target location
        #Check if agent is about to bump into a wall
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == 'turnright':
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.R)
        elif action == 'turnleft':
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.L)
        elif action == 'moveforward':
            print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            agent.moveforward()
        elif action == "eat":
            items = self.list_things_at(agent.location, tclass=Food)
            if len(items) != 0:
                if agent.eat(items[0]):
                    print('{} ate {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "drink":
            items = self.list_things_at(agent.location, tclass=Water)
            if len(items) != 0:
                if agent.drink(items[0]):
                    print('{} drank {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but to prevent killing our cute dog, we will stop before itself - when there is no more food or water'''
        no_edibles = not any(isinstance(thing, Food) or isinstance(thing, Water) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles



# %% [markdown]
# Now that our park is ready for the 2D motion of our energetic dog, lets test it!

# %%
park = Park2D(5,5, color={'EnergeticBlindDog': (200,0,0), 'Water': (0, 200, 200), 'Food': (230, 115, 40)})
dog = EnergeticBlindDog(program)
dogfood = Food()
water = Water()
park.add_thing(dog, [0,0])
park.add_thing(dogfood, [1,2])
park.add_thing(water, [0,1])
morewater = Water()
morefood = Food()
park.add_thing(morewater, [2,4])
park.add_thing(morefood, [4,3])
print("dog started at [0,0], facing down. Let's see if he found any food or water!")
park.run(20)

# %% [markdown]
#
#
#
#
#
#
#
#
# ## Wumpus Environment

# %%
from ipythonblocks import BlockGrid
from aima.agents import *

color = {"Breeze": (225, 225, 225),
        "Pit": (0,0,0),
        "Gold": (253, 208, 23),
        "Glitter": (253, 208, 23),
        "Wumpus": (43, 27, 23),
        "Stench": (128, 128, 128),
        "Explorer": (0, 0, 255),
        "Wall": (44, 53, 57)
        }

def program(percepts):
    '''Returns an action based on it's percepts'''
    print(percepts)
    return input()

w = WumpusEnvironment(program, 7, 7)         
grid = BlockGrid(w.width, w.height, fill=(123, 234, 123))

def draw_grid(world):
    global grid
    grid[:] = (123, 234, 123)
    for x in range(0, len(world)):
        for y in range(0, len(world[x])):
            if len(world[x][y]):
                grid[y, x] = color[world[x][y][-1].__class__.__name__]

def step():
    global grid, w
    draw_grid(w.get_world())
    grid.show()
    w.step()


# %%
step()

# %%
