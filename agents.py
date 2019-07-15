'''
This file holds the agents.

'''

import random, copy, collections
from objects import Object

# ______________________________________________________________________________

class Agent(Object):
    '''
    An Agent is a subclass of Object with one required slot, .program, which should hold a function that takes one
    argument, the percept, and returns an action. (What counts as a percept or action will depend on the specific
    environment in which the agent exists.)  Note that 'program' is a slot, not a method.  If it were a method, then
    the program could 'cheat' and look at aspects of the agent.  It's not supposed to do that: the program can only
    look at the percepts.  An agent program that needs a model of the world (and of the agent itself) will have to
    build and maintain its own model.  There is an optional slots, .performance, which is a number giving the
    performance measure of the agent in its environment.
    '''

    def __init__(self):
        def program(percept):
            return raw_input('Percept=%s; action? ' % percept)

        self.program = program
        self.alive = True

        self.bump = False
        self.holding = []

        self.performance = 0

        if program is None or not isinstance(program, collections.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

    blocker = True

def DebugAgent(agent):
    '''
    Wrap the agent's program to print its input and output. This will let you see what the agent is doing in the
    environment.
    '''

    # Mark: Do we just replace the agent parent class with DebugAgent to enable printing?
    old_program = agent.program
    def new_program(percept):
        action = old_program(percept)
        print('%s perceives %s and does %s' % (agent, percept, action))
        return action
    agent.program = new_program
    return agent
#______________________________________________________________________________

class XYAgent(Agent):
    holding = []
    heading = (1, 0)

class RandomXYAgent(XYAgent):
    "An agent that chooses an action at random, ignoring all percepts."

    def __init__(self, actions):
        Agent.__init__(self)
        self.program = lambda percept: random.choice(actions)

def NewRandomXYAgent(debug=False):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward']))
    else:
        return RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward'])
    #return RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Grab', 'Release'])

class RandomReflexAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, actions):
        Agent.__init__(self)
        self.actions = actions

        def program(percept):
            if percept[0] == 'Dirty':
                return "Grab"
            else:
                return random.choice(actions)
        self.program = program

def NewRandomReflexAgent(debug=False):
    "If the cell is dirty, Grab the dirt; otherwise, randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(RandomReflexAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward']))
    else:
        return RandomReflexAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward'])

class SimpleReflexAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, rules, interpret_input):
        Agent.__init__(self)
        def program(percept):
            state = interpret_input(percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        self.program = program

class ReflexAgentWithState(XYAgent):
    '''This agent takes action based on the percept and state. [Fig. 2.16]'''

    def __init__(self, rules, udpate_state):
        Agent.__init__(self)
        state, action = None, None
        def program(percept):
            state = update_state(state, action, percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        self.program = program

#______________________________________________________________________________
