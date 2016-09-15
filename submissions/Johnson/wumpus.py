# ______________________________________________________________________________
# The Wumpus World


class Gold(Thing):

    def __eq__(self, rhs):
        '''All Gold are equal'''
        return rhs.__class__ == Gold
    pass

class Bump(Thing):
    pass

class Glitter(Thing):
    pass

class Pit(Thing):
    pass

class Breeze(Thing):
    pass


class Arrow(Thing):
    pass

class Scream(Thing):
    pass


class Wumpus(Agent):
    screamed = False
    pass

class Stench(Thing):
    pass

class Explorer(Agent):
    holding = []
    has_arrow = True
    killed_by = ""
    direction = Direction("right")

    def can_grab(self, thing):
        '''Explorer can only grab gold'''
        return thing.__class__ == Gold


class WumpusEnvironment(XYEnvironment):
    pit_probability = 0.2 #Probability to spawn a pit in a location. (From Chapter 7.2)
    #Room should be 4x4 grid of rooms. The extra 2 for walls
    def __init__(self, agent_program, width=6, height=6):
        super(WumpusEnvironment, self).__init__(width, height)
        self.init_world(agent_program)

    def init_world(self, program):
        '''Spawn items to the world based on probabilities from the book'''

        "WALLS"
        self.add_walls()

        "PITS"
        for x in range(self.x_start, self.x_end):
            for y in range(self.y_start, self.y_end):
                if random.random() < self.pit_probability:
                    self.add_thing(Pit(), (x,y), True)
                    self.add_thing(Breeze(), (x - 1,y), True)
                    self.add_thing(Breeze(), (x,y - 1), True)
                    self.add_thing(Breeze(), (x + 1,y), True)
                    self.add_thing(Breeze(), (x,y + 1), True)

        "WUMPUS"
        w_x, w_y = self.random_location_inbounds(exclude = (1,1))
        self.add_thing(Wumpus(lambda x: ""), (w_x, w_y), True)
        self.add_thing(Stench(), (w_x - 1, w_y), True)
        self.add_thing(Stench(), (w_x + 1, w_y), True)
        self.add_thing(Stench(), (w_x, w_y - 1), True)
        self.add_thing(Stench(), (w_x, w_y + 1), True)

        "GOLD"
        self.add_thing(Gold(), self.random_location_inbounds(exclude = (1,1)), True)
        #self.add_thing(Gold(), (2,1), True)  Making debugging a whole lot easier

        "AGENT"
        self.add_thing(Explorer(program), (1,1), True)

    def get_world(self, show_walls = True):
        '''returns the items in the world'''
        result = []
        x_start,y_start = (0,0)  if show_walls else (1,1)
        x_end,y_end = (self.width, self.height) if show_walls else (self.width - 1, self.height - 1)
        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x,y)))
            result.append(row)
        return result

    def percepts_from(self, agent, location, tclass = Thing):
        '''Returns percepts from a given location, and replaces some items with percepts from chapter 7.'''
        thing_percepts = {
            Gold: Glitter(),
            Wall: Bump(),
            Wumpus: Stench(),
            Pit: Breeze()
            }
        '''Agents don't need to get their percepts'''
        thing_percepts[agent.__class__] = None

        '''Gold only glitters in its cell'''
        if location != agent.location:
            thing_percepts[Gold] = None


        result = [thing_percepts.get(thing.__class__, thing) for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]
        return result if len(result) else [None]

    def percept(self, agent):
        '''Returns things in adjacent (not diagonal) cells of the agent.
            Result format: [Left, Right, Up, Down, Center / Current location]'''
        x,y = agent.location
        result = []
        result.append(self.percepts_from(agent, (x - 1,y)))
        result.append(self.percepts_from(agent, (x + 1,y)))
        result.append(self.percepts_from(agent, (x,y - 1)))
        result.append(self.percepts_from(agent, (x,y + 1)))
        result.append(self.percepts_from(agent, (x,y)))

        '''The wumpus gives out a a loud scream once it's killed.'''
        wumpus = [thing for thing in self.things if isinstance(thing, Wumpus)]
        if len(wumpus) and not wumpus[0].alive and not wumpus[0].screamed:
            result[-1].append(Scream())
            wumpus[0].screamed = True

        return result

    def execute_action(self, agent, action):
        '''Modify the state of the environment based on the agent's actions
            Performance score taken directly out of the book'''

        if isinstance(agent, Explorer) and self.in_danger(agent):
            return

        agent.bump = False
        if action == 'TurnRight':
            agent.direction = agent.direction + Direction.R
            agent.performance -= 1
        elif action == 'TurnLeft':
            agent.direction = agent.direction + Direction.L
            agent.performance -= 1
        elif action == 'Forward':
            agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
            agent.performance -= 1
        elif action == 'Grab':
            things = [thing for thing in self.list_things_at(agent.location)
                    if agent.can_grab(thing)]
            if len(things):
                print("Grabbing", things[0].__class__.__name__)
                if len(things):
                    agent.holding.append(things[0])
            agent.performance -= 1
        elif action == 'Climb':
            if agent.location == (1,1): #Agent can only climb out of (1,1)
                agent.performance += 1000 if Gold() in agent.holding else 0
                self.delete_thing(agent)
        elif action == 'Shoot':
            '''The arrow travels straight down the path the agent is facing'''
            if agent.has_arrow:
                arrow_travel = agent.direction.move_forward(agent.location)
                while(self.is_inbounds(arrow_travel)):
                    wumpus = [thing for thing in self.list_things_at(arrow_travel)
                              if isinstance(thing, Wumpus)]
                    if len(wumpus):
                        wumpus[0].alive = False
                        break
                    arrow_travel = agent.direction.move_forward(agent.location)
                agent.has_arrow = False

    def in_danger(self, agent):
        '''Checks if Explorer is in danger (Pit or Wumpus), if he is, kill him'''
        for thing in self.list_things_at(agent.location):
            if isinstance(thing, Pit) or (isinstance(thing, Wumpus) and thing.alive):
                agent.alive = False
                agent.performance -= 1000
                agent.killed_by = thing.__class__.__name__
                return True
        return False

    def is_done(self):
        '''The game is over when the Explorer is killed
            or if he climbs out of the cave only at (1,1)'''
        explorer = [agent for agent in self.agents if isinstance(agent, Explorer) ]
        if len(explorer):
                if explorer[0].alive:
                       return False
                else:
                    print("Death by {} [-1000].".format(explorer[0].killed_by))
        else:
            print("Explorer climbed out {}."
                  .format("with Gold [+1000]!" if Gold() not in self.things else "without Gold [+0]"))
        return True

    #Almost done. Arrow needs to be implemented
