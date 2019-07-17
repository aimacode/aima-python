'''
This is hosted on github.

This is heavily based on the example from Artificial Intelligence: A Modern Approach located here:
http://aima.cs.berkeley.edu/python/agents.html
http://aima.cs.berkeley.edu/python/agents.py
'''

import inspect
from utils import *
import random, copy
from functools import partial

# import my files
from agents import *
from objects import *
from display import *
from comms import *

'''Implement Agents and Environments (Chapters 1-2).

The class hierarchies are as follows:

Object ## A physical object that can exist in an environment
    Agent
        RandomReflexAgent
        ...
    Dirt
    Wall
        DeadCell
    Fire
    ...

Environment ## An environment holds objects, runs simulations
    XYEnvironment
        VacuumEnvironment

EnvFrame ## A graphical representation of the Environment

'''







class Environment:
    """
    Abstract class representing an Environment.  'Real' Environment classes inherit from this. Your Environment will
    typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .objects and .agents (which is a subset of .objects). Each agent has a .performance
    slot, initialized to 0.  Each object has a .location slot, even though some environments may not need this.
    """

    def __init__(self,):
        self.objects = []
        self.agents = []
        self.perceptors = {}
        self.communicator = Communicator(self)

    # Mark: What does this do?  It isn't checked in the Environment class's add_object.
    object_classes = [] ## List of classes that can go into environment

    def percept(self, agent):
        agentpercept = {}  # initialize the percept dictionary
        for per in agent.perceptorTypes:  # for each perceptor in agent
            # calculate the percept value for the perceptor and append to the percept dictionary
            agentpercept.update(self.perceptors[per.__name__].percept(agent))
        return agentpercept

    def execute_action(self, agent, action):
        "Change the world to reflect this action. Override this."
        raise NotImplementedError

    def default_location(self, obj):
        "Default location to place a new object with unspecified location"
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
        '''Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do.  If there are interactions between them, you'll need to
        override this method.'''
        if not self.is_done():
            # for each agent
            # run agent.program with the agent's preception as an input
            # agent's perception = Env.precept(agent)

            for a in self.agents:
                a.percepts = self.percept(a)

            # TODO: Implement comms
            self.communicator.run_comms(self.agents)

#            comms = {}
#            for to_agent in self.agents:
#                agents_seen = self.communicator.get_comms_network(to_agent)
#                comms[to_agent] = [self.communicator.communicate(percepts[from_agent],from_agent,to_agent) for from_agent in agents_seen]

#            for to_agent in self.agents:
#                percepts = to_agent.merge_comms

            actions = [agent.program(agent.percepts)
                       for agent in self.agents]

            # for each agent-action pair, have the environment process the actions
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)

            # process any external events
            self.exogenous_change()

    def run(self, steps=1000):
        for step in range(steps): # Run the Environment for given number of time steps.
            if self.is_done(): return
            self.step()

    def add_object(self, obj, location=None):
        '''Add an object to the environment, setting its location. Also keep track of objects that are agents.
        Shouldn't need to override this.'''
        obj.location = location or self.default_location(obj)
        # Mark: ^^ unsure about this line, lazy evaluation means it will only process if location=None?
        # Add the new Object to self.objects
        self.objects.append(obj)
        # If the object is an Agent, add it to self.agents and initialize performance parameter
        if isinstance(obj, Agent):
            obj.performance = 0
            self.add_perceptor_for_agent(obj)
            self.agents.append(obj)
        return obj

    def add_perceptor_for_agent(self, agent):
        for pertype in agent.perceptorTypes: # for each type of perceptor for the agent
            if not [p for p in self.perceptors.values() if isinstance(p, pertype)]: # if the perceptor doesn't exist yet
                print('creating perceptor of type %s' % pertype.__name__)
                self.perceptors[pertype.__name__] = pertype(self) # add the name:perceptor pair to the dictionary


class XYEnvironment(Environment):
    '''This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.  Agents
    perceive objects within a radius.  Each agent in the environment
    has a .location slot which should be a location such as (0, 1),
    and a .holding slot, which should be a list of objects that are
    held '''

    #robot_images = {(1,0):'img/robot-right.gif',(-1,0):'img/robot-left.gif',(0,-1):'img/robot-up.gif',(0,1):'img/robot-down.gif'}

    def __init__(self, width=10, height=10):
        # set all of the initial conditions with the update function
        self.width = width
        self.height = height
        Environment.__init__(self)

    def objects_of_type(self, cls):
        # Use a list comprehension to return a list of all objects of type cls
        return [obj for obj in self.objects if isinstance(obj, cls)]

    def objects_at(self, location):
        "Return all objects exactly at a given location."
        return [obj for obj in self.objects if obj.location == location]

    def find_at(self, cls, loc):
        return [o for o in self.objects_at(loc) if isinstance(o, cls)]

    def objects_near(self, location, radius):
        "Return all objects within radius of location."
        radius2 = radius * radius # square radius instead of taking the square root for faster processing
        return [obj for obj in self.objects
                if distance2(location[0], location[1], obj.location[0], obj.location[1]) <= radius2]

#    def percept(self, agent): # Unused, currently at default settings
#        "By default, agent perceives objects within radius r."
#        return [self.object_percept(obj, agent)
#                for obj in self.objects_near(agent, 3)]

    def execute_action(self, agent, action):
        # TODO: Add stochasticity
        # TODO: Add actions on objects e.g. Grab(Target)

        # The world processes actions on behalf of an agent.
        # Agents decide what to do, but the Environment class actually processes the behavior.
        #
        # Implemented actions are 'TurnRignt', 'TurnLeft', 'Forward', 'Grab', 'Release'
        if action == 'TurnRight':
            # decrement the heading by -90° by getting the previous index of the headings array
            agent.heading = self.turn_heading(agent.heading, -1)
        elif action == 'TurnLeft':
            # increment the heading by +90° by getting the next index of the headings array
            agent.heading = self.turn_heading(agent.heading, +1)
        elif action == 'Forward':
            # move the Agent in the facing direction by adding the heading vector to the Agent location
            self.move_to(agent, vector_add(agent.heading, agent.location))
        elif action == 'Grab':
            # check to see if any objects at the Agent's location are grabbable by the Agent
            objs = [obj for obj in self.objects_at(agent.location)
                if (obj != agent and obj.is_grabbable(agent))]
            # if so, pick up all grabbable objects and add them to the holding array
            if objs:
                agent.holding += objs
                for o in objs:
                    # set the location of the Object = the Agent instance carrying the Object
                    # by setting the location to an object instead of a tuple, we can now detect
                    # when to remove if from the display.  This may be useful in other ways, if
                    # the object needs to know who it's holder is
                    o.location = agent
                    if isinstance(o,Dirt): agent.performance += 100
        elif action == 'Release':
            # drop an objects being held by the Agent.
            if agent.holding:
                # restore the location parameter to add the object back to the display
                agent.holding.pop().location = agent.location

    def default_location(self, obj):
        # If no location is specified, set the location to be a random location in the Environment.
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, obj, destination):
        "Move an object to a new location."
        # Currently move_to assumes that the object is only moving a single cell at a time
        # e.g. agent.location + agent.heading => (x,y) + (0,1)
        #
        # The function finds all objects at the destination that have the blocker flag set.
        # If there are none, move to the destination

        obstacles = [os for os in self.objects_at(destination) if os.blocker]
        if not obstacles:
            obj.location = destination


    def add_object(self, obj, location=(1, 1)):
        Environment.add_object(self, obj, location)

        obj.holding = []
        obj.held = None

        return obj

    def add_walls(self):
        "Put walls around the entire perimeter of the grid."
        for x in range(self.width-1):
            self.add_object(Wall(), (x, 0))
            self.add_object(Wall(), (x+1, self.height-1))
        for y in range(self.height-1):
            self.add_object(Wall(), (0, y+1))
            self.add_object(Wall(), (self.width-1, y))

    def turn_heading(self, heading, inc,
                 headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]):
        "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
        return headings[(headings.index(heading) + inc) % len(headings)]

#______________________________________________________________________________
## Vacuum environment

class VacuumEnvironment(XYEnvironment):
    '''The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken.'''
    def __init__(self, width=10, height=10):
        XYEnvironment.__init__(self, width, height)
        self.add_walls()

    object_classes = []

#    def percept(self, agent):
#        status =  if_(self.find_at(Dirt, agent.location), 'Dirty', 'Clean')
#        bump = if_(agent.bump, 'Bump', 'None')
#        dirts = [obj.location for obj in self.objects_of_type(Dirt) if not isinstance(obj.location, Agent)]
#        return (status, bump, dirts, agent.location, agent.heading)

    def execute_action(self, agent, action):
        if action == 'Suck':
            if self.find_at(Dirt, agent.location):
                agent.performance += 100
        agent.performance -= 1
        XYEnvironment.execute_action(self, agent, action)

    def exogenous_change(self):
        pass

def NewVacuumEnvironment(width=10, height=10, config=None):
    e = VacuumEnvironment(width=width, height=height)
    # Generate walls with dead cells in the center
    if config==None:
        pass
    elif config=='center walls':
        for x in range(int(e.width/2-5),int(e.width/2+5)):
            for y in range(int(e.height/2-5),int(e.height/2+5)):
                if ((x == int(e.width/2-5)) or (x == int(e.width/2+4)) or
                    (y == int(e.height/2-5)) or (y == int(e.height/2+4))):
                    e.add_object(Wall(), (x,y))
                else:
                    e.add_object(DeadCell(), (x,y))
    elif config=='full dirt':
        # Fill a square area with dirt
        if False:
            for x in range(0,e.width):
                for y in range(0,e.height):
                    if e.find_at(Wall,(x,y)): e.add_object(Dirt(),location=(x,y))
    elif config=='center walls w/ random dirt and fire':
        for x in range(int(e.width/2-5),int(e.width/2+5)):
            for y in range(int(e.height/2-5),int(e.height/2+5)):
                if ((x == int(e.width/2-5)) or (x == int(e.width/2+4)) or
                    (y == int(e.height/2-5)) or (y == int(e.height/2+4))):
                    e.add_object(Wall(), (x,y))
                else:
                    e.add_object(DeadCell(), (x,y))

        # adds custom behavior to the exogenous_chage() method to avoid creating a new class
        # is that correct?  should we just create a new class?

        def exogenous_dirt(self):
            if random.uniform(0, 1) < 1.0:
                loc = (random.randrange(self.width), random.randrange(self.height))
                if not (self.find_at(Dirt, loc) or self.find_at(Wall, loc)):
                    self.add_object(Dirt(), loc)

        def exogenous_fire(self):
            fs = self.objects_of_type(Fire)

            if fs:
                for f in fs:
                    if f.t == 0:
                        f.destroy()
                        self.objects.remove(f)
                    else:
                        f.t -= 1
                        if random.uniform(0, 1) < 0.21:
                            emptyCells = [(x, y) for x in range(f.location[0] - 1, f.location[0] + 2)
                                          for y in range(f.location[1] - 1, f.location[1] + 2)
                                          if not self.objects_at((x, y))]
                            if emptyCells: self.add_object(Fire(), random.choice(emptyCells))
            else:  # if there is no fire
                for i in range(5):
                    for i in range(10):  # try 10 times, would do while, but that could get stuck
                        loc = (random.randrange(1, self.width), random.randrange(1, self.width))
                        if not self.objects_at(loc):
                            self.add_object(Fire(), loc)
                            break

        old_exogenous_chage = e.exogenous_change
        def new_exogenous_change(self):
            old_exogenous_chage()
            exogenous_dirt(self)
            exogenous_fire(self)

        e.exogenous_change = MethodType(new_exogenous_change, e)

    return e
#______________________________________________________________________________

def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    '''See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of
    each one for steps. Return a list of (agent, average-score) tuples.'''
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs)))
            for A in AgentFactories]

def test_agent(AgentFactory, steps, envs):
    "Return the mean score of running an agent in each of the envs, for steps"
    total = 0
    i = 0
    for env in envs:
        i+=1
        with Timer(name='Simulation Timer - Agent=%s' % i, format='%.4f'):
            agent = AgentFactory()
            env.add_object(agent)
            env.run(steps)
            total += agent.performance
    return float(total)/len(envs)

#______________________________________________________________________________

def test1():
    e = NewVacuumEnvironment(width=20,height=20,config="center walls w/ random dirt and fire")
    ef = EnvFrame(e,cellwidth=30)

    # Create agents on left wall
    for i in range(1,19):
        e.add_object(NewRandomReflexAgent(debug=False),location=(1,i)).id = i

    ef.configure_display()
    ef.run()
    ef.mainloop()

def test2():
    EnvFactory = partial(NewVacuumEnvironment,width=10,height=10,config="full dirt")
    AgentFactory = partial(NewRandomReflexAgent, debug=False)
    print(compare_agents(EnvFactory, [AgentFactory]*2, n=10, steps=1000))

def test3():
    e = NewVacuumEnvironment(width=20,height=20,config="center walls w/ random dirt and fire")
    ef = EnvFrame(e,cellwidth=30)

    # Create agents on left wall
    for i in range(1,19):
        e.add_object(GreedyAgent(), location=(1,i)).id = i

    ef.configure_display()
    ef.run()
    ef.mainloop()

def main():
    # set a seed to provide repeatable outcomes each run
    random.seed(0) # set seed to None to remove the seed and have different outcomes

    test3()

if __name__ == "__main__":
    # execute only if run as a script
    main()
