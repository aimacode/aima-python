import agents as ag
import envgui as gui
# change this line ONLY to refer to your project
import submissions.aardvark.vacuum2 as v2

# ______________________________________________________________________________
# Vacuum environment

class Dirt(ag.Thing):
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
        return [ag.Wall, Dirt,
                # ReflexVacuumAgent, RandomVacuumAgent,
                # TableDrivenVacuumAgent, ModelBasedVacuumAgent
                ]

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

# Launch a Text-Based Environment
print('Two Cells, Agent on Left:')
v = VacuumEnvironment(4, 3)
v.add_thing(Dirt(), (1, 1))
v.add_thing(Dirt(), (2, 1))
a = v2.HW2Agent()
a = ag.TraceAgent(a)
v.add_thing(a, (1, 1))
t = gui.EnvTUI(v)
t.mapImageNames({
    ag.Wall: '#',
    # Floor: 'images/floor.png',
    Dirt: '@',
    ag.Agent: 'V',
})
t.step(0)
t.list_things(Dirt)
t.step(4)
if len(t.env.get_things(Dirt)) > 0:
    t.list_things(Dirt)
else:
    print('All clean!')

# Check to continue
if input('Do you want to continue [y/N]? ') != 'y':
    exit(0)

# Repeat, but put Agent on the Right
print('Two Cells, Agent on Right:')
v = VacuumEnvironment(4, 3)
v.add_thing(Dirt(), (1, 1))
v.add_thing(Dirt(), (2, 1))
a = v2.HW2Agent()
a = ag.TraceAgent(a)
v.add_thing(a, (2, 1))
t = gui.EnvTUI(v)
t.mapImageNames({
    ag.Wall: '#',
    # Floor: 'images/floor.png',
    Dirt: '@',
    ag.Agent: 'V',
})
t.step(0)
t.list_things(Dirt)
t.step(4)
if len(t.env.get_things(Dirt)) > 0:
    t.list_things(Dirt)
else:
    print('All clean!')

# Check to continue
if input('Do you want to continue [y/N]? ') != 'y':
    exit(0)

# Repeat, but put Agent on the Right
print('Two Cells, Agent on Top:')
v = VacuumEnvironment(3, 4)
v.add_thing(Dirt(), (1, 1))
v.add_thing(Dirt(), (1, 2))
a = v2.HW2Agent()
a = ag.TraceAgent(a)
v.add_thing(a, (1, 1))
t = gui.EnvTUI(v)
t.mapImageNames({
    ag.Wall: '#',
    # Floor: 'images/floor.png',
    Dirt: '@',
    ag.Agent: 'V',
})
t.step(0)
t.list_things(Dirt)
t.step(4)
if len(t.env.get_things(Dirt)) > 0:
    t.list_things(Dirt)
else:
    print('All clean!')

# Check to continue
if input('Do you want to continue [y/N]? ') != 'y':
    exit(0)

v = VacuumEnvironment(6, 4)
a = v2.HW2Agent()
a = ag.TraceAgent(a)
loc = v.random_location_inbounds()
v.add_thing(a, location=loc)
v.scatter_things(Dirt)
g = gui.EnvGUI(v, 'Vaccuum')
c = g.getCanvas()
c.mapImageNames({
    ag.Wall: 'images/wall.jpg',
    # Floor: 'images/floor.png',
    Dirt: 'images/dirt.png',
    ag.Agent: 'images/vacuum.png',
})
c.update()
g.mainloop()