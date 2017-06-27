import random
from agents import Direction
from agents import Agent
from agents import ReflexVacuumAgent, ModelBasedVacuumAgent, TrivialVacuumEnvironment, compare_agents,\
                   RandomVacuumAgent


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


def test_Agent():
    def constant_prog(percept):
        return percept
    agent = Agent(constant_prog)
    result = agent.program(5)
    assert result == 5
