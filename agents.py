"""Implement Agents and Environments (Chapters 1-2).

The class hierarchies are as follows:

Object ## A physical object that can exist in an environment
    Agent
        Wumpus
        RandomAgent
        ReflexVacuumAgent
        ...
    Dirt
    Wall
    ...
    
Environment ## An environment holds objects, runs simulations
    XYEnvironment
        VacuumEnvironment
        WumpusEnvironment

EnvGUI ## A window with a graphical representation of the Environment

EnvToolbar ## contains buttons for controlling EnvGUI

EnvCanvas ## Canvas to display the environment of an EnvGUI

"""

# TO DO:
# Implement grabbing correctly.
# When an object is grabbed, does it still have a location?
# What if it is released?
# What if the grabbed or the grabber is deleted?
# What if the grabber moves?
#
# Speed control in GUI does not have any effect -- fix it.

from utils import *
import random, copy

#______________________________________________________________________________


class Object (object):
    """This represents any physical object that can appear in an Environment.
    You subclass Object to get the objects you want.  Each object can have a
    .__name__  slot (used for output only)."""
    def __repr__(self):
        return '<%s>' % getattr(self, '__name__', self.__class__.__name__)

    def is_alive(self):
        """Objects that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state.  Subclasses should override."""
        print "I don't know how to show_state."

    def display(self, canvas, x, y, width, height):
        # Do we need this?
        """Display an image of this Object on the canvas."""
        pass

class Agent (Object):
    """An Agent is a subclass of Object with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.) 
    Note that 'program' is a slot, not a method.  If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts.  An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slots, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self):
        self.program = self.make_agent_program()
        self.alive = True
        self.bump = False

    def make_agent_program(self):
        
        def program(percept):
            return raw_input('Percept=%s; action? ' % percept)
        return program

    def can_grab(self, obj):
        """Returns True if this agent can grab this object.
        Override for appropriate subclasses of Agent and Object."""
        return False
    
def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program
    def new_program(percept):
        action = old_program(percept)
        print '%s perceives %s and does %s' % (agent, percept, action)
        return action
    agent.program = new_program
    return agent

#______________________________________________________________________________

class TableDrivenAgent (Agent):
    """This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it you provide a table to the constructor. [Fig. 2.7]"""
    
    def __init__(self, table):
        "Supply as table a dictionary of all {percept_sequence:action} pairs."
        ## The agent program could in principle be a function, but because
        ## it needs to store state, we make it a callable instance of a class.
        self.table = table
        super(TableDrivenAgent, self).__init__()

    def make_agent_program(self):
        table = self.table
        percepts = []
        def program(percept):
            percepts.append(percept)
            action = table.get(tuple(percepts))
            return action
        return program


class RandomAgent (Agent):
    "An agent that chooses an action at random, ignoring all percepts."

    def __init__(self, actions):
        self.actions = actions
        super(RandomAgent, self).__init__()

    def make_agent_program(self):
        actions = self.actions
        return lambda percept: random.choice(actions)


#______________________________________________________________________________

loc_A, loc_B = (0, 0), (1, 0) # The two locations for the Vacuum world

class ReflexVacuumAgent (Agent):
    "A reflex agent for the two-state vacuum environment. [Fig. 2.8]"

    def __init__(self):
        super(ReflexVacuumAgent, self).__init__()

    def make_agent_program(self):
        def program((location, status)):
            if status == 'Dirty': return 'Suck'
            elif location == loc_A: return 'Right'
            elif location == loc_B: return 'Left'
        return program

def RandomVacuumAgent():
    "Randomly choose one of the actions from the vacuum environment."
    return RandomAgent(['Right', 'Left', 'Suck', 'NoOp'])


def TableDrivenVacuumAgent():
    "[Fig. 2.3]"
    table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Clean'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             # ...
             ((loc_A, 'Clean'), (loc_A, 'Clean'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_A, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             # ...
             }
    return TableDrivenAgent(table)


class ModelBasedVacuumAgent (Agent):
    "An agent that keeps track of what locations are clean or dirty."

    def __init__(self):
        self.model = {loc_A: None, loc_B: None}
        super(ModelBasedVacuumAgent, self).__init__()

    def make_agent_program(self):
        model = self.model
        def program((location, status)):
            "Same as ReflexVacuumAgent, except if everything is clean, do NoOp"
            model[location] = status ## Update the model here
            if model[loc_A] == model[loc_B] == 'Clean': return 'NoOp'
            elif status == 'Dirty': return 'Suck'
            elif location == loc_A: return 'Right'
            elif location == loc_B: return 'Left'
        return program

#______________________________________________________________________________


class Environment (object):
    """Abstract class representing an Environment.  'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .objects and .agents (which is a subset
    of .objects). Each agent has a .performance slot, initialized to 0.
    Each object has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.objects = []
        self.agents = []

    def object_classes(self):
        return [] ## List of classes that can go into environment

    def percept(self, agent):
	"Return the percept that the agent sees at this point. Override this."
        abstract

    def execute_action(self, agent, action):
        "Change the world to reflect this action. Override this."
        abstract

    def default_location(self, object):
	"Default location to place a new object with unspecified location."
        return None

    def exogenous_change(self):
	"If there is spontaneous change in the world, override this."
	pass

    def is_done(self):
        "By default, we're done when we can't find a live agent."
        for agent in self.agents:
            if agent.is_alive(): return False
        return True

    def step(self):
	"""Run the environment for one time step. If the
	actions and exogenous changes are independent, this method will
	do.  If there are interactions between them, you'll need to
	override this method."""
	if not self.is_done():
            actions = [agent.program(self.percept(agent))
                       for agent in self.agents]
            for (agent, action) in zip(self.agents, actions):
		self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
	"""Run the Environment for given number of time steps."""
	for step in range(steps):
            if self.is_done(): return
            self.step()

    def list_objects_at(self, location, oclass=Object):
        "Return all objects exactly at a given location."
        return [obj for obj in self.objects
                if obj.location == location and isinstance(obj, oclass)]
    
    def some_objects_at(self, location, oclass=Object):
        """Return true if at least one of the objects at location
        is an instance of class oclass.

        'Is an instance' in the sense of 'isinstance',
        which is true if the object is an instance of a subclass of oclass."""

        return self.list_objects_at(location, oclass) != []

    def add_object(self, obj, location=None):
	"""Add an object to the environment, setting its location. Also keep
	track of objects that are agents.  Shouldn't need to override this."""

	obj.location = location or self.default_location(obj)
	self.objects.append(obj)
	if isinstance(obj, Agent):
            obj.performance = 0
            self.agents.append(obj)
	return self

    def delete_object(self, obj):
        """Remove an object from the environment."""
        try:
            self.objects.remove(obj)
        except ValueError, e:
            print e
            print "  in Environment delete_object"
            print "  Object to be removed: %s at %s" % (obj, obj.location)
            trace_list("  from list", self.objects)
        if obj in self.agents:
            self.agents.remove(obj)


def trace_list (name, objlist):
    ol_list = [(obj, obj.location) for obj in objlist]
    print "%s: %s" % (name, ol_list)

class XYEnvironment (Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.

    Agents perceive objects within a radius.  Each agent in the
    environment has a .location slot which should be a location such
    as (0, 1), and a .holding slot, which should be a list of objects
    that are held."""

    def __init__(self, width=10, height=10):
        super(XYEnvironment, self).__init__()
        self.width = width
        self.height = height
        #update(self, objects=[], agents=[], width=width, height=height)
        self.observers = []
        
    def objects_near(self, location, radius):
        "Return all objects within radius of location."
        radius2 = radius * radius
        return [obj for obj in self.objects
                if distance2(location, obj.location) <= radius2]

    def percept(self, agent):
        "By default, agent perceives objects within radius r."
        ### Error below: objects_near requires also a radius argument
        return [self.object_percept(obj, agent)
                for obj in self.objects_near(agent)] ### <- error

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'TurnRight':
            agent.heading = self.turn_heading(agent.heading, -1)
        elif action == 'TurnLeft':
            agent.heading = self.turn_heading(agent.heading, +1)
        elif action == 'Forward':
            self.move_to(agent, vector_add(agent.heading, agent.location))
#         elif action == 'Grab':
#             objs = [obj for obj in self.list_objects_at(agent.location)
#                     if agent.can_grab(obj)]
#             if objs:
#                 agent.holding.append(objs[0])
        elif action == 'Release':
            if agent.holding:
                agent.holding.pop()

    def object_percept(self, obj, agent): #??? Should go to object?
        "Return the percept for this object."
        return obj.__class__.__name__

    def default_location(self, object):
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, obj, destination):
        "Move an object to a new location."

        # Bumped?
        obj.bump = self.some_objects_at(destination, Obstacle)

        if not obj.bump:
            # Move object and report to observers
            obj.location = destination
            for o in self.observers:
                o.object_moved(obj)
        
    def add_object(self, obj, location=(1, 1)):
        super(XYEnvironment, self).add_object(obj, location)
        obj.holding = []
        obj.held = None
        # self.objects.append(obj) # done in Environment!
        # Report to observers
        for obs in self.observers:
            obs.object_added(obj)

    def delete_object(self, obj):
        super(XYEnvironment, self).delete_object(obj)
        # Any more to do?  Object holding anything or being held?
        for obs in self.observers:
            obs.object_deleted(obj)
    
    def add_walls(self):
        "Put walls around the entire perimeter of the grid."
        for x in range(self.width):
            self.add_object(Wall(), (x, 0))
            self.add_object(Wall(), (x, self.height-1))
        for y in range(self.height):
            self.add_object(Wall(), (0, y))
            self.add_object(Wall(), (self.width-1, y))

    def add_observer(self, observer):
        """Adds an observer to the list of observers.  
        An observer is typically an EnvGUI.
        
        Each observer is notified of changes in move_to and add_object,
        by calling the observer's methods object_moved(obj, old_loc, new_loc)
        and object_added(obj, loc)."""
        self.observers.append(observer)
        
    def turn_heading(self, heading, inc, headings=orientations):
        "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
        return headings[(headings.index(heading) + inc) % len(headings)]  

class Obstacle (Object):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    pass

class Wall (Obstacle):
    pass

#______________________________________________________________________________
## Vacuum environment 

class Dirt (Object):
    pass
    
class VacuumEnvironment (XYEnvironment):
    """The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken."""

    def __init__(self, width=10, height=10):
        super(VacuumEnvironment, self).__init__(width, height)
        self.add_walls()

    def object_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        status = if_(self.some_objects_at(agent.location, Dirt),
                     'Dirty', 'Clean')
        bump = if_(agent.bump, 'Bump', 'None')
        return (status, bump)

    def execute_action(self, agent, action):
        if action == 'Suck':
            dirt_list = self.list_objects_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_object(dirt)
        else:
            super(VacuumEnvironment, self).execute_action(agent, action)

        if action != 'NoOp':
            agent.performance -= 1

class TrivialVacuumEnvironment (Environment):

    """This environment has two locations, A and B. Each can be Dirty
    or Clean.  The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super(TrivialVacuumEnvironment, self).__init__()
        self.status = {loc_A:random.choice(['Clean', 'Dirty']),
                       loc_B:random.choice(['Clean', 'Dirty'])}

    def object_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, 
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]
    
    def percept(self, agent):
        "Returns the agent's location, and the location status (Dirty/Clean)."
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, object):
        "Agents start in either location at random."
        return random.choice([loc_A, loc_B])

#______________________________________________________________________________

class SimpleReflexAgent (Agent):
    """This agent takes action based solely on the percept. [Fig. 2.13]"""

    def __init__(self, rules, interpret_input):
        self.rules = rules
        self.interpret_input = interpret_input
        super(SimpleReflexAgent, self).__init__()

    def make_agent_program(self):
        rules = self.rules
        interpret_input = self.interpret_input
        def program(percept):
            state = interpret_input(percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        return program

class ReflexAgentWithState (Agent):
    """This agent takes action based on the percept and state. [Fig. 2.16]"""

    def __init__(self, rules, update_state):
        self.rules = rules
        self.update_state = update_state
        super(ReflexAgentWithState, self).__init__()

    def make_agent_program(self):
        rules = self.rules
        update_state = self.update_state
        state = None
        action = None
        def program(percept):
            state = update_state(state, action, percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        return program

def rule_match(state, rules):
    "Find the first rule that matches state."
    for rule in rules:
        if rule.matches(state):
            return rule

#______________________________________________________________________________
## The Wumpus World

class Gold (Object): pass
class Pit (Object): pass
class Arrow (Object): pass
class Wumpus (Agent): pass
class Explorer (Agent): pass

class WumpusEnvironment(XYEnvironment):

    def __init__(self, width=10, height=10):
        super(WumpusEnvironment, self).__init__(width, height)
        self.add_walls()

    def object_classes(self):
        return [Wall, Gold, Pit, Arrow, Wumpus, Explorer]

    ## Needs a lot of work ...

    
#______________________________________________________________________________

def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    """See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of 
    each one for steps. Return a list of (agent, average-score) tuples."""
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs))) 
            for A in AgentFactories]

def test_agent(AgentFactory, steps, envs):
    "Return the mean score of running an agent in each of the envs, for steps"
    total = 0
    for env in envs:
        agent = AgentFactory()
        env.add_object(agent)
        env.run(steps)
        total += agent.performance
    return float(total)/len(envs)

#_________________________________________________________________________

_docex = """
a = ReflexVacuumAgent()
a.program
a.program((loc_A, 'Clean')) ==> 'Right'
a.program((loc_B, 'Clean')) ==> 'Left'
a.program((loc_A, 'Dirty')) ==> 'Suck'
a.program((loc_A, 'Dirty')) ==> 'Suck'

e = TrivialVacuumEnvironment()
e.add_object(TraceAgent(ModelBasedVacuumAgent()))
e.run(5)

## Environments, and some agents, are randomized, so the best we can
## give is a range of expected scores.  If this test fails, it does
## not necessarily mean something is wrong.
envs = [TrivialVacuumEnvironment() for i in range(100)]
def testv(A): return test_agent(A, 4, copy.deepcopy(envs)) 
testv(ModelBasedVacuumAgent)
(7 < _ < 11) ==> True
testv(ReflexVacuumAgent)
(5 < _ < 9) ==> True
testv(TableDrivenVacuumAgent)
(2 < _ < 6) ==> True
testv(RandomVacuumAgent)
(0.5 < _ < 3) ==> True
"""

#______________________________________________________________________________
# GUI - Graphical User Interface for Environments
# If you do not have Tkinter installed, either get a new installation of Python
# (Tkinter is standard in all new releases), or delete the rest of this file
# and muddle through without a GUI.

import Tkinter as tk

class EnvGUI (tk.Tk, object):

    def __init__(self, env, title = 'AIMA GUI', cellwidth=50, n=10):

        # Initialize window
        
        super(EnvGUI, self).__init__()
        self.title(title)

        # Create components
        
        canvas = EnvCanvas(self, env, cellwidth, n)
        toolbar = EnvToolbar(self, env, canvas)
        for w in [canvas, toolbar]:
            w.pack(side="bottom", fill="x", padx="3", pady="3")


class EnvToolbar (tk.Frame, object):

    def __init__(self, parent, env, canvas):
        super(EnvToolbar, self).__init__(parent, relief='raised', bd=2)

        # Initialize instance variables
        
        self.env = env
        self.canvas = canvas
        self.running = False
        self.speed = 1.0

        # Create buttons and other controls
        
        for txt, cmd in [('Step >', self.env.step), ('Run >>', self.run),
                         ('Stop [ ]', self.stop),
                         ('List objects', self.list_objects),
                         ('List agents', self.list_agents)]:
            tk.Button(self, text=txt, command=cmd).pack(side='left')

        tk.Label(self, text='Speed').pack(side='left')
        scale = tk.Scale(self, orient='h',
                         from_=(1.0), to=10.0, resolution=1.0,
                         command=self.set_speed)
        scale.set(self.speed)
        scale.pack(side='left')

    def run(self):
        print 'run'
        self.running = True
        self.background_run()

    def stop(self):
        print 'stop'
        self.running = False

    def background_run(self):
        if self.running:
            self.env.step()
            # ms = int(1000 * max(float(self.speed), 0.5))
            #ms = max(int(1000 * float(self.delay)), 1)
            delay_sec = 1.0 / max(self.speed, 1.0) # avoid division by zero
            ms = int(1000.0 * delay_sec)  # seconds to milliseconds
            self.after(ms, self.background_run)
        
    def list_objects(self):
        print "Objects in the environment:"
        for obj in self.env.objects:
            print "%s at %s" % (obj, obj.location)

    def list_agents(self):
        print "Agents in the environment:"
        for agt in self.env.agents:
            print "%s at %s" % (agt, agt.location)

    def set_speed(self, speed):
        self.speed = float(speed)
    
