import agents as ag
import envgui as gui
import importlib
import traceback

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

# # Launch a Text-Based Environment
# print('Two Cells, Agent on Left:')
# v = VacuumEnvironment(4, 3)
# v.add_thing(Dirt(), (1, 1))
# v.add_thing(Dirt(), (2, 1))
# a = v2.HW2Agent()
# a = ag.TraceAgent(a)
# v.add_thing(a, (1, 1))
# t = gui.EnvTUI(v)
# t.mapImageNames({
#     ag.Wall: '#',
#     Dirt: '@',
#     ag.Agent: 'V',
# })
# t.step(0)
# t.list_things(Dirt)
# t.step(4)
# if len(t.env.get_things(Dirt)) > 0:
#     t.list_things(Dirt)
# else:
#     print('All clean!')
#
# # Check to continue
# if input('Do you want to continue [y/N]? ') != 'y':
#     exit(0)
# else:
#     print('----------------------------------------')

class MyException(Exception):
    pass

roster = ['Ban','Becker','Blue','Capps','Conklin','Dickenson','Fritz',
          'Haller','Hawley','Hess','Johnson','Karman','Kinley','LaMartina',
          'McLean','Miles','Ottenlips','Porter','Sery','VanderKallen',
          'aardvark','zzzsolutions',
          ]

submissions = {}
scores = {}
def testVacuum(student, label, w=4, h=3,
               dloc=[(1,1),(2,1)],
               vloc=(1,1),
               limit=6,
               points=5.0):
    print(student + ', ' + label)
    v = VacuumEnvironment(w, h)
    dStart = len(dloc)
    for loc in dloc:
        v.add_thing(Dirt(), loc)
    try:
        a = submissions[student]()
        a = ag.TraceAgent(a)
        v.add_thing(a, vloc)
        t = gui.EnvTUI(v)
        t.mapImageNames({
            ag.Wall: '#',
            Dirt: '@',
            ag.Agent: 'V',
        })
        t.step(0)
        t.list_things(Dirt)
        t.step(limit)
    except:
        traceback.print_exc()

    dFinish = len(t.env.get_things(Dirt))
    if dFinish > 0:
        t.list_things(Dirt)
    else:
        print('All clean!')

    newPoints = points * (dStart - dFinish) / dStart
    scores[student].append(newPoints)
    print(student + ' scores ' + str(newPoints))

    # Check to continue
    if input('Continue testing ' + student + ' [Y/n]? ') == 'n':
        print('----------------------------------------')
        raise MyException
        return False
    else:
        print('----------------------------------------')
        return True

for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.vacuum2')
        submissions[student] = mod.HW2Agent
    except:
        pass

for student in submissions:
    scores[student] = []
    try:
        testVacuum(student, 'Two Cells, Agent on Left:', points=35)
        testVacuum(student, 'Two Cells, Agent on Right:',
                          vloc=(2,1), points=35)
        testVacuum(student, 'Two Cells, Agent on Top:',
                          w=3, h=4, dloc=[(1,1), (1,2)], vloc=(1,1), points=2.5)
        testVacuum(student, 'Two Cells, Agent on Bottom:',
                          w=3, h=4, dloc=[(1,1), (1,2)], vloc=(1,2), points=2.5)
        testVacuum(student, 'Five Cells, Agent on Left:',
                          w=7, h=3, dloc=[(2,1), (4,1)], vloc=(1,1),
                          limit=12, points=2.5)
        testVacuum(student, 'Five Cells, Agent near Right:',
                          w=7, h=3, dloc=[(2,1), (3,1)], vloc=(4,1),
                          limit=12, points=2.5)
        testVacuum(student, 'Five Cells, Agent on Top:',
                          w=3, h=7, dloc=[(1,2), (1,4)], vloc=(1,1),
                          limit=12, points=2.5)
        testVacuum(student, 'Five Cells, Agent Near Bottom:',
                          w=3, h=7, dloc=[(1,2), (1,3)], vloc=(1,4),
                          limit=12, points=2.5)
        testVacuum(student, '5x4 Grid, Agent in Top Left:',
                          w=7, h=6, dloc=[(1,4), (2,2), (3, 3), (4,1), (5,2)],
                          vloc=(1,1), limit=34, points=2.5)
        testVacuum(student, '5x4 Grid, Agent near Bottom Right:',
                          w=7, h=6, dloc=[(1,3), (2,2), (3, 4), (4,1), (5,2)],
                          vloc=(4, 3), limit=34, points=2.5)
        testVacuum(student, '8x10 Grid, Agent near Top Right:',
                          w=10, h=12, dloc=[(4, 1), (7, 2), (2, 3), (5, 4), (8, 5),
                                            (3, 6), (6, 7), (1, 8), (4, 9), (7, 10)],
                          vloc=(1,1), limit=108, points=2.5)
        testVacuum(student, '8x10 Grid, Agent in Bottom Left:',
                          w=10, h=12, dloc=[(4, 1), (7, 2), (2, 3), (5, 4), (8, 5),
                                            (3, 6), (6, 7), (1, 8), (4, 9), (7, 10)],
                          vloc=(4, 3), limit=108, points=2.5)
    except:
        pass

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')

# v = VacuumEnvironment(6, 3)
# a = v2.HW2Agent()
# a = ag.TraceAgent(a)
# loc = v.random_location_inbounds()
# v.add_thing(a, location=loc)
# v.scatter_things(Dirt)
# g = gui.EnvGUI(v, 'Vaccuum')
# c = g.getCanvas()
# c.mapImageNames({
#     ag.Wall: 'images/wall.jpg',
#     # Floor: 'images/floor.png',
#     Dirt: 'images/dirt.png',
#     ag.Agent: 'images/vacuum.png',
# })
# c.update()
# g.mainloop()