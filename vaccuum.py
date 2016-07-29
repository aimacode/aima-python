import agents as ag
import envgui as gui
import random

# ______________________________________________________________________________

loc_A, loc_B = (1, 1), (2, 1)  # The two locations for the Vacuum world


def RandomVacuumAgent():
    "Randomly choose one of the actions from the vacuum environment."
    p = ag.RandomAgentProgram(['Right', 'Left', 'Up', 'Down', 'Suck', 'NoOp'])
    return ag.Agent(p)


def TableDrivenVacuumAgent():
    "[Figure 2.3]"
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
    p = ag.TableDrivenAgentProgram(table)
    return ag.Agent()


def ReflexVacuumAgent():
    "A reflex agent for the two-state vacuum environment. [Figure 2.8]"
    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'
    return ag.Agent(program)


def ModelBasedVacuumAgent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    model = {loc_A: None, loc_B: None}

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        location, status = percept
        model[location] = status  # Update the model here
        if model[loc_A] == model[loc_B] == 'Clean':
            return 'NoOp'
        elif status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'
    return ag.Agent(program)

# ______________________________________________________________________________
# Vacuum environment

class Dirt(ag.Thing):
    pass

class Floor(ag.Thing):
    pass


class VacuumEnvironment(ag.XYEnvironment):

    """The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken."""

    def __init__(self, width=4, height=3):
        super(VacuumEnvironment, self).__init__(width, height)
        self.add_walls()

    def thing_classes(self):
        return [ag.Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        status = ('Dirty' if self.some_things_at(
            agent.location, Dirt) else 'Clean')
        bump = ('Bump' if agent.bump else'None')
        return (bump, status)

    def execute_action(self, agent, action):
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
        else:
            super(VacuumEnvironment, self).execute_action(agent, action)

        if action != 'NoOp':
            agent.performance -= 1


class TrivialVacuumEnvironment(VacuumEnvironment):

    """This environment has two locations, A and B. Each can be Dirty
    or Clean.  The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super(TrivialVacuumEnvironment, self).__init__()
        # self.status = {loc_A: random.choice(['Clean', 'Dirty']),
        #                loc_B: random.choice(['Clean', 'Dirty'])}
        self.add_thing(Dirt(), self.random_location_inbounds())

    # def thing_classes(self):
    #     return [ag.Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
    #             TableDrivenVacuumAgent, ModelBasedVacuumAgent]
    #
    def percept(self, agent):
        "Returns the agent's location, and the location status (Dirty/Clean)."
        status = ('Dirty' if self.some_things_at(
            agent.location, Dirt) else 'Clean')
        return (agent.location, status)
    #
    # def execute_action(self, agent, action):
    #     """Change agent's location and/or location's status; track performance.
    #     Score 10 for each dirt cleaned; -1 for each move."""
    #     if action == 'Right':
    #         agent.location = loc_B
    #         agent.performance -= 1
    #     elif action == 'Left':
    #         agent.location = loc_A
    #         agent.performance -= 1
    #     elif action == 'Suck':
    #         if self.status[agent.location] == 'Dirty':
    #             agent.performance += 10
    #         self.status[agent.location] = 'Clean'
    #
    # def default_location(self, thing):
    #     "Agents start in either location at random."
    #     return random.choice([loc_A, loc_B])


# _________________________________________________________________________

# >>> a = ReflexVacuumAgent()
# >>> a.program((loc_A, 'Clean'))
# 'Right'
# >>> a.program((loc_B, 'Clean'))
# 'Left'
# >>> a.program((loc_A, 'Dirty'))
# 'Suck'
# >>> a.program((loc_A, 'Dirty'))
# 'Suck'
#
# >>> e = TrivialVacuumEnvironment()
# >>> e.add_thing(ModelBasedVacuumAgent())
# >>> e.run(5)

# Produces text-based status output
# v = TrivialVacuumEnvironment()
# a = ModelBasedVacuumAgent()
# a = ag.TraceAgent(a)
# v.add_thing(a)
# v.run(20)

# Launch GUI
v = TrivialVacuumEnvironment()
# v = VacuumEnvironment(5, 4)
# a = ModelBasedVacuumAgent()
a = RandomVacuumAgent()
a = ag.TraceAgent(a)
#v.add_thing(Floor(), location=(1, 1))
#v.add_thing(Floor(), location=(2, 1))
# v.add_thing(Dirt(), location=(1, 1))
# v.add_thing(Dirt(), location=(2, 1))
v.add_thing(a, location=(1, 1))
g = gui.EnvGUI(v, 'Vaccuum')
c = g.getCanvas()
c.mapImageNames({
    ag.Wall: 'images/wall.jpg',
    Floor: 'images/floor.png',
    Dirt: 'images/Dirt.png',
    ag.Agent: 'images/vacuum.png',
})
c.update()
g.mainloop()
