import random

import pytest

from agents import (ReflexVacuumAgent, ModelBasedVacuumAgent, TrivialVacuumEnvironment, compare_agents,
                    RandomVacuumAgent, TableDrivenVacuumAgent, TableDrivenAgentProgram, RandomAgentProgram,
                    SimpleReflexAgentProgram, ModelBasedReflexAgentProgram, Wall, Gold, Explorer, Thing, Bump, Glitter,
                    WumpusEnvironment, Pit, VacuumEnvironment, Dirt, Direction, Agent)

# random seed may affect the placement
# of things in the environment which may
# lead to failure of tests. Please change
# the seed if the tests are failing with
# current changes in any stochastic method
# function or variable.
random.seed(9)

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


def test_RandomAgentProgram():
    # create a list of all the actions a Vacuum cleaner can perform
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
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_RandomVacuumAgent():
    # create an object of the RandomVacuumAgent
    agent = RandomVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_TableDrivenAgent():
    random.seed(10)
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
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'}

    # create an program and then an object of the TableDrivenAgent
    program = TableDrivenAgentProgram(table)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # initializing some environment status
    environment.status = {loc_A: 'Dirty', loc_B: 'Dirty'}
    # add agent to the environment
    environment.add_thing(agent)

    # run the environment by single step everytime to check how environment evolves using TableDrivenAgentProgram
    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Dirty'}

    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Dirty'}

    environment.run(steps=1)
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ReflexVacuumAgent():
    # create an object of the ReflexVacuumAgent
    agent = ReflexVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


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
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ModelBasedReflexAgentProgram():
    class Rule:

        def __init__(self, state, action):
            self.__state = state
            self.action = action

        def matches(self, state):
            return self.__state == state

    loc_A = (0, 0)
    loc_B = (1, 0)

    # create rules for a two-state Vacuum Environment
    rules = [Rule((loc_A, "Dirty"), "Suck"), Rule((loc_A, "Clean"), "Right"),
             Rule((loc_B, "Dirty"), "Suck"), Rule((loc_B, "Clean"), "Left")]

    def update_state(state, action, percept, model):
        return percept

    # create a program and then an object of the ModelBasedReflexAgentProgram class
    program = ModelBasedReflexAgentProgram(rules, update_state, None)
    agent = Agent(program)
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_ModelBasedVacuumAgent():
    # create an object of the ModelBasedVacuumAgent
    agent = ModelBasedVacuumAgent()
    # create an object of TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_TableDrivenVacuumAgent():
    # create an object of the TableDrivenVacuumAgent
    agent = TableDrivenVacuumAgent()
    # create an object of the TrivialVacuumEnvironment
    environment = TrivialVacuumEnvironment()
    # add agent to the environment
    environment.add_thing(agent)
    # run the environment
    environment.run()
    # check final status of the environment
    assert environment.status == {(1, 0): 'Clean', (0, 0): 'Clean'}


def test_compare_agents():
    environment = TrivialVacuumEnvironment
    agents = [ModelBasedVacuumAgent, ReflexVacuumAgent]

    result = compare_agents(environment, agents)
    performance_ModelBasedVacuumAgent = result[0][1]
    performance_ReflexVacuumAgent = result[1][1]

    # The performance of ModelBasedVacuumAgent will be at least as good as that of
    # ReflexVacuumAgent, since ModelBasedVacuumAgent can identify when it has
    # reached the terminal state (both locations being clean) and will perform
    # NoOp leading to 0 performance change, whereas ReflexVacuumAgent cannot
    # identify the terminal state and thus will keep moving, leading to worse
    # performance compared to ModelBasedVacuumAgent.
    assert performance_ReflexVacuumAgent <= performance_ModelBasedVacuumAgent


def test_TableDrivenAgentProgram():
    table = {(('foo', 1),): 'action1',
             (('foo', 2),): 'action2',
             (('bar', 1),): 'action3',
             (('bar', 2),): 'action1',
             (('foo', 1), ('foo', 1),): 'action2',
             (('foo', 1), ('foo', 2),): 'action3'}
    agent_program = TableDrivenAgentProgram(table)
    assert agent_program(('foo', 1)) == 'action1'
    assert agent_program(('foo', 2)) == 'action3'
    assert agent_program(('invalid percept',)) is None


def test_Agent():
    def constant_prog(percept):
        return percept

    agent = Agent(constant_prog)
    result = agent.program(5)
    assert result == 5


def test_VacuumEnvironment():
    # initialize Vacuum Environment
    v = VacuumEnvironment(6, 6)
    # get an agent
    agent = ModelBasedVacuumAgent()
    agent.direction = Direction(Direction.R)
    v.add_thing(agent)
    v.add_thing(Dirt(), location=(2, 1))

    # check if things are added properly
    assert len([x for x in v.things if isinstance(x, Wall)]) == 20
    assert len([x for x in v.things if isinstance(x, Dirt)]) == 1

    # let the action begin!
    assert v.percept(agent) == ("Clean", "None")
    v.execute_action(agent, "Forward")
    assert v.percept(agent) == ("Dirty", "None")
    v.execute_action(agent, "TurnLeft")
    v.execute_action(agent, "Forward")
    assert v.percept(agent) == ("Dirty", "Bump")
    v.execute_action(agent, "Suck")
    assert v.percept(agent) == ("Clean", "None")
    old_performance = agent.performance
    v.execute_action(agent, "NoOp")
    assert old_performance == agent.performance


def test_WumpusEnvironment():
    def constant_prog(percept):
        return percept

    # initialize Wumpus Environment
    w = WumpusEnvironment(constant_prog)

    # check if things are added properly
    assert len([x for x in w.things if isinstance(x, Wall)]) == 20
    assert any(map(lambda x: isinstance(x, Gold), w.things))
    assert any(map(lambda x: isinstance(x, Explorer), w.things))
    assert not any(map(lambda x: not isinstance(x, Thing), w.things))

    # check that gold and wumpus are not present on (1,1)
    assert not any(map(lambda x: isinstance(x, Gold) or isinstance(x, WumpusEnvironment), w.list_things_at((1, 1))))

    # check if w.get_world() segments objects correctly
    assert len(w.get_world()) == 6
    for row in w.get_world():
        assert len(row) == 6

    # start the game!
    agent = [x for x in w.things if isinstance(x, Explorer)][0]
    gold = [x for x in w.things if isinstance(x, Gold)][0]
    pit = [x for x in w.things if isinstance(x, Pit)][0]

    assert not w.is_done()

    # check Walls
    agent.location = (1, 2)
    percepts = w.percept(agent)
    assert len(percepts) == 5
    assert any(map(lambda x: isinstance(x, Bump), percepts[0]))

    # check Gold
    agent.location = gold.location
    percepts = w.percept(agent)
    assert any(map(lambda x: isinstance(x, Glitter), percepts[4]))
    agent.location = (gold.location[0], gold.location[1] + 1)
    percepts = w.percept(agent)
    assert not any(map(lambda x: isinstance(x, Glitter), percepts[4]))

    # check agent death
    agent.location = pit.location
    assert w.in_danger(agent)
    assert not agent.alive
    assert agent.killed_by == Pit.__name__
    assert agent.performance == -1000

    assert w.is_done()


def test_WumpusEnvironmentActions():
    random.seed(9)
    def constant_prog(percept):
        return percept

    # initialize Wumpus Environment
    w = WumpusEnvironment(constant_prog)

    agent = [x for x in w.things if isinstance(x, Explorer)][0]
    gold = [x for x in w.things if isinstance(x, Gold)][0]
    pit = [x for x in w.things if isinstance(x, Pit)][0]

    agent.location = (1, 1)
    assert agent.direction.direction == "right"
    w.execute_action(agent, 'TurnRight')
    assert agent.direction.direction == "down"
    w.execute_action(agent, 'TurnLeft')
    assert agent.direction.direction == "right"
    w.execute_action(agent, 'Forward')
    assert agent.location == (2, 1)

    agent.location = gold.location
    w.execute_action(agent, 'Grab')
    assert agent.holding == [gold]

    agent.location = (1, 1)
    w.execute_action(agent, 'Climb')
    assert not any(map(lambda x: isinstance(x, Explorer), w.things))

    assert w.is_done()


if __name__ == "__main__":
    pytest.main()
