import random
from agents import Direction
from agents import Agent
from agents import ReflexVacuumAgent, ModelBasedVacuumAgent, TrivialVacuumEnvironment, compare_agents,\
                   RandomVacuumAgent, TableDrivenVacuumAgent, TableDrivenAgentProgram, RandomAgentProgram, \
		   SimpleReflexAgentProgram, rule_match


random.seed("aima-python")


def test_move_forward():
    d = Direction("up")
    l1 = d.move_forward((0, 0))
    assert l1 == (0, -1)

    d = Direction(Direction.R)
    l1 = d.move_forward((0, 0))
    assert l1 == (1, 0)

    d = Direction(Direction.D)
    l1 = d.move_forward((0, 0))
    assert l1 == (0, 1)

    d = Direction("left")
    l1 = d.move_forward((0, 0))
    assert l1 == (-1, 0)

    l2 = d.move_forward((1, 0))
    assert l2 == (0, 0)


def test_add():
    d = Direction(Direction.U)
    l1 = d + "right"
    l2 = d + "left"
    assert l1.direction == Direction.R
    assert l2.direction == Direction.L

    d = Direction("right")
    l1 = d.__add__(Direction.L)
    l2 = d.__add__(Direction.R)
    assert l1.direction == "up"
    assert l2.direction == "down"

    d = Direction("down")
    l1 = d.__add__("right")
    l2 = d.__add__("left")
    assert l1.direction == Direction.L
    assert l2.direction == Direction.R

    d = Direction(Direction.L)
    l1 = d + Direction.R
    l2 = d + Direction.L
    assert l1.direction == Direction.U
    assert l2.direction == Direction.D

def test_RandomAgentProgram() :
    #create a list of all the actions a vacuum cleaner can perform
    list = ['Right', 'Left', 'Suck', 'NoOp']
    # create a program and then an object of the RandomAgentProgram
    program = RandomAgentProgram(list)
    
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean' , (0, 0): 'Clean'}

def test_RandomVacuumAgent() :
    # create an object of the RandomVacuumAgent
    agent = RandomVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}


def test_TableDrivenAgent():
    loc_A, loc_B = (0, 0), (1, 0)
    # table defining all the possible states of the agent
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

    # create an program and then an object of the TableDrivenAgent
    program = TableDrivenAgentProgram(table)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # initializing some environment status
    environment.status = {loc_A:'Dirty', loc_B:'Dirty'}
    # add agent to the environment
    environment.add_thing(agent)

    # run the environment by single step everytime to check how environment evolves using TableDrivenAgentProgram
    environment.run(steps = 1)
    assert environment.status == {(1,0): 'Clean', (0,0): 'Dirty'}

    environment.run(steps = 1)
    assert environment.status == {(1,0): 'Clean', (0,0): 'Dirty'}

    environment.run(steps = 1)
    assert environment.status == {(1,0): 'Clean', (0,0): 'Clean'}


def test_ReflexVacuumAgent() :
    # create an object of the ReflexVacuumAgent
    agent = ReflexVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}

def test_SimpleReflexAgentProgram():
    class Rule:
        
        def __init__(self, state, action):
            self.__state = state
            self.action = action
            
        def matches(self, state):
            return self.__state == state
        
    loc_A = (0, 0)
    loc_B = (1, 0)
    
    # create rules for a two state Vacuum Environment
    rules = [Rule((loc_A, "Dirty"), "Suck"), Rule((loc_A, "Clean"), "Right"),
            Rule((loc_B, "Dirty"), "Suck"), Rule((loc_B, "Clean"), "Left")]
    
    def interpret_input(state):
        return state
    
    # create a program and then an object of the SimpleReflexAgentProgram
    program = SimpleReflexAgentProgram(rules, interpret_input) 
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}


def test_ModelBasedVacuumAgent() :
    # create an object of the ModelBasedVacuumAgent
    agent = ModelBasedVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}


def test_TableDrivenVacuumAgent() :
    # create an object of the TableDrivenVacuumAgent
    agent = TableDrivenVacuumAgent()
    # create an object of the TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0):'Clean', (0, 0):'Clean'}


def test_compare_agents() :
    environment = TrivialVacuumEnvironment
    agents = [ModelBasedVacuumAgent, ReflexVacuumAgent]

    result = compare_agents(environment, agents)
    performance_ModelBasedVacummAgent = result[0][1]
    performance_ReflexVacummAgent = result[1][1]

    # The performance of ModelBasedVacuumAgent will be at least as good as that of
    # ReflexVacuumAgent, since ModelBasedVacuumAgent can identify when it has
    # reached the terminal state (both locations being clean) and will perform
    # NoOp leading to 0 performance change, whereas ReflexVacuumAgent cannot
    # identify the terminal state and thus will keep moving, leading to worse
    # performance compared to ModelBasedVacuumAgent.
    assert performance_ReflexVacummAgent <= performance_ModelBasedVacummAgent


def test_TableDrivenAgentProgram():
    table = {(('foo', 1),): 'action1',
             (('foo', 2),): 'action2',
             (('bar', 1),): 'action3',
             (('bar', 2),): 'action1',
             (('foo', 1), ('foo', 1),): 'action2',
             (('foo', 1), ('foo', 2),): 'action3',
             }
    agent_program = TableDrivenAgentProgram(table)
    assert agent_program(('foo', 1)) == 'action1'
    assert agent_program(('foo', 2)) == 'action3'
    assert agent_program(('invalid percept',)) == None


def test_Agent():
    def constant_prog(percept):
        return percept
    agent = Agent(constant_prog)
    result = agent.program(5)
    assert result == 5
