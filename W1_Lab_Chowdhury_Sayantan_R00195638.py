# %%
## Question B
print("+ Assignment Part B +".center(80, '-'))

from agents import loc_A, loc_B, TableDrivenVacuumAgent, TrivialVacuumEnvironment
def runEnvironment(agent, env, n_runs):
    print("No of runs:", n_runs)
    env.add_thing(agent)
    for step in range(n_runs):
        env.step()
        print(f"Run: {step+1}: Env. status: (0, 0): {env.status[loc_A]}, " \
              f"(1, 0): {env.status[loc_B]} " \
              f"Agent Performance: {agent.performance}")
    print("\n")

if __name__ == '__main__':
    agent = TableDrivenVacuumAgent()
    environment = TrivialVacuumEnvironment()
    runEnvironment(agent, environment, 2)

    agent = TableDrivenVacuumAgent()
    environment = TrivialVacuumEnvironment()
    runEnvironment(agent, environment, 4)

    agent = TableDrivenVacuumAgent()
    environment = TrivialVacuumEnvironment()
    runEnvironment(agent, environment, 8)


answer4 = """
Here the agent's performace metrics is not sufficient to determine the
quality of work done, as the agent is only awarded when it sucks the dirt.

Optimal status is when both the tiles are dirty, i.e. the agent gets to
earn a total of 19 points over the given lifetime
"""

answer5 = """
Least optimal status would be when the environment doesn't have any dirt to
start with, the agent's movement to perceive the environment causes it to
lose a point.
"""

print(answer4, "\n", answer5)
# %%
## Question C
print("+ Assignment Part C +".center(80, '-'))
answer2to5 = """
 Agent:  Farmer
+----------------------------------------------------+--+-----------------+------------------+
|                       Percepts                     |  |                 |                  |
+----------------------------------------------------+--+-----------------+------------------+
|           loc_A        |  |           loc_B        |  | Actions         | Agent's Location |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
| Chicken | Fox   | Feed |  |         |       |      |  | Go with Chicken | loc_A            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
|         | Fox   | Feed |  | Chicken |       |      |  | Return          | loc_B            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
|         | Fox   | Feed |  | Chicken |       |      |  | Go With Fox     | loc_A            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
|         |       | Feed |  | Chicken | Fox   |      |  | Go With Chicken | loc_B            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
| Chicken |       | Feed |  |         | Fox   |      |  | Go With Feed    | loc_A            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
| Chicken |       |      |  |         | Fox   | Feed |  | Return          | loc_B            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
| Chicken |       |      |  |         | Fox   | Feed |  | Go With Chicken | loc_A            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
|         |       |      |  | Chicken | Fox   | Feed |  |                 | loc_B            |
+---------+-------+------+--+---------+-------+------+--+-----------------+------------------+
"""
print(answer2to5)

from agents import  Environment, Agent, Thing

# TableDrivenAgentProgram,

class Chicken(Thing):
    def __repr__(self):
        return 'Chicken'

class Fox(Thing):
    def __repr__(self):
        return 'Fox'

class Feed(Thing):
    def __repr__(self):
        return 'Feed'

loc_A = 'loc_A'
loc_B = 'loc_B'

x = []


def TableDrivenAgentProgram(table):
    """
    [Figure 2.7]
    This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs.
    """
    percepts = []

    def program(percept):
        percepts.append(percept)
        global x
        x = percepts
        action = table.get(tuple(percepts))
        return action

    return program


def TableDrivenFarmerAgent():

    """Tabular approach towards vacuum world as mentioned in [Figure 2.3]
    >>> agent = TableDrivenVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True

    Taken from aima-python github code and edited to fit requirement
    """
    table = {
      ((loc_A, Chicken(), Fox(), Feed()),)                                                                                                                                                     :   'GoWithChicken',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()))                                                                                                                                  :   'Return',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()), (loc_A, Fox(), Feed()))                                                                                                          :   'GoWithFox',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()), (loc_A, Fox(), Feed()), (loc_B, Fox(), Chicken()))                                                                               :   'GoWithChicken',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()), (loc_A, Fox(), Feed()), (loc_B, Fox(), Chicken()), (loc_A, Chicken(), Feed()))                                                   :   'GoWithFeed',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()), (loc_A, Fox(), Feed()), (loc_B, Fox(), Chicken()), (loc_A, Chicken(), Feed()), (loc_B, Fox(), Feed()))                           :   'Return',
      ((loc_A, Chicken(), Fox(), Feed()), (loc_B, Chicken()), (loc_A, Fox(), Feed()), (loc_B, Fox(), Chicken()), (loc_A, Chicken(), Feed()), (loc_B, Fox(), Feed()), (loc_A, Chicken()))       :   'GoWithChicken'
    }
    ## Sorting the tuple of tuples in the key of the table dictionary
    table = {tuple(tuple([i[0], *sorted(map(repr, i[1:]))]) for i in k) : v for k, v in table.items()}
    return Agent(TableDrivenAgentProgram(table))

class FarmersDilemmaEnvironment(Environment):
    """This environment has two locations, A and B, separated by a water body
    in between. Location A has a Farmer, a Fox, a chicken and a bag of feed
    for the chicken. Location B is empty. The agent travels between A and B
    perceives the passenger status and carries the farmer and one other
    passenger at a time.

    Taken from aima-python github code and edited to fit requirement
    """

    def __init__(self):
        super().__init__()
        self.status = {loc_A: [],
                       loc_B: []}

    def init_env(self):
        self.status = {loc_A: [repr(thing) for thing in fdenv.things if not isinstance(thing, Agent) and thing.location == loc_A],
            loc_B: [repr(thing) for thing in fdenv.things if not isinstance(thing, Agent) and thing.location == loc_B]}

    def thing_classes(self):
        return [TableDrivenFarmerAgent, Chicken, Fox, Feed]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Passenger Names)."""
        return agent.location, *sorted(self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 when finished; -1 for each move."""
        print(f"{self.status}".ljust(55, ' '), f"Farmer's Location: {agent.location}".ljust(25, ' '), f"Action: {action}\n")
        loc_toggler = lambda : loc_B if agent.location == loc_A else loc_A
        if self.status[loc_A]:
            if action[:6] == 'GoWith':
                self.status[agent.location].remove(action[6:])
                agent.location = loc_toggler()
                self.status[agent.location].append(action[6:])
                agent.performance -= 1

            elif action == 'Return':
                agent.location = loc_toggler()
                agent.performance -= 1

        if not self.status[loc_A]:
            agent.performance += 10
            agent.alive = False

    def default_location(self, thing):
        """Agent start in location A."""
        return loc_A

if __name__ == '__main__':
    chicken = Chicken()
    feed = Feed()
    fox = Fox()
    agent = TableDrivenFarmerAgent()
    fdenv = FarmersDilemmaEnvironment()
    fdenv.add_thing(agent)
    fdenv.add_thing(chicken, location=loc_A)
    fdenv.add_thing(feed, location=loc_A)
    fdenv.add_thing(fox, location=loc_A)
    fdenv.init_env()
    fdenv.run()

# %%
