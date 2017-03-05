"""Planning (Chapters 10-11)
"""

import itertools
from utils import Expr, expr, first
from logic import FolKB

class PDLL:
    """
    PDLL used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial_state, actions, goal_test):
        self.kb = FolKB(initial_state)
        self.actions = actions
        self.goal_test_func = goal_test

    def goal_test(self):
        return self.goal_test_func(self.kb)

    def act(self, action):
        """
        Performs the action given as argument.
        Note that action is an Expr like expr('Remove(Glass, Table)') or expr('Eat(Sandwich)')
        """
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.kb, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        list_action(self.kb, args)

class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PDDL.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negated literals.
    Example:
    precond_pos = [expr("Human(person)"), expr("Hungry(Person)")]
    precond_neg = [expr("Eaten(food)")]
    effect_add = [expr("Eaten(food)")]
    effect_rem = [expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), [precond_pos, precond_neg], [effect_add, effect_rem])
    """

    def __init__(self, action, precond, effect):
        self.name = action.op
        self.args = action.args
        self.precond_pos = precond[0]
        self.precond_neg = precond[1]
        self.effect_add = effect[0]
        self.effect_rem = effect[1]

    def __call__(self, kb, args):
        return self.act(kb, args)

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propositional symbol"""
        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i in range(len(self.args)):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""
        # check for positive clauses
        for clause in self.precond_pos:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        # check for negative clauses
        for clause in self.precond_neg:
            if self.substitute(clause, args) in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's kb"""
        # check if the preconditions are satisfied
        if not self.check_precond(kb, args):
            raise Exception("Action pre-conditions not satisfied")
        # remove negative literals
        for clause in self.effect_rem:
            kb.retract(self.substitute(clause, args))
        # add positive literals
        for clause in self.effect_add:
            kb.tell(self.substitute(clause, args))


def air_cargo():
    init = [expr('At(C1, SFO)'),
            expr('At(C2, JFK)'),
            expr('At(P1, SFO)'),
            expr('At(P2, JFK)'),
            expr('Cargo(C1)'),
            expr('Cargo(C2)'),
            expr('Plane(P1)'),
            expr('Plane(P2)'),
            expr('Airport(JFK)'),
            expr('Airport(SFO)')]

    def goal_test(kb):
        required = [expr('At(C1 , JFK)'), expr('At(C2 ,SFO)')]
        for q in required:
            if kb.ask(q) is False:
                return False
        return True

    ## Actions
    #  Load
    precond_pos = [expr("At(c, a)"), expr("At(p, a)"), expr("Cargo(c)"), expr("Plane(p)"), expr("Airport(a)")]
    precond_neg = []
    effect_add = [expr("In(c, p)")]
    effect_rem = [expr("At(c, a)")]
    load = Action(expr("Load(c, p, a)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #  Unload
    precond_pos = [expr("In(c, p)"), expr("At(p, a)"), expr("Cargo(c)"), expr("Plane(p)"), expr("Airport(a)")]
    precond_neg = []
    effect_add = [expr("At(c, a)")]
    effect_rem = [expr("In(c, p)")]
    unload = Action(expr("Unload(c, p, a)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #  Fly
    #  Used 'f' instead of 'from' because 'from' is a python keyword and expr uses eval() function
    precond_pos = [expr("At(p, f)"), expr("Plane(p)"), expr("Airport(f)"), expr("Airport(to)")]
    precond_neg = []
    effect_add = [expr("At(p, to)")]
    effect_rem = [expr("At(p, f)")]
    fly = Action(expr("Fly(p, f, to)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDLL(init, [load, unload, fly], goal_test)


def spare_tire():
    init = [expr('Tire(Flat)'),
            expr('Tire(Spare)'),
            expr('At(Flat, Axle)'),
            expr('At(Spare, Trunk)')]

    def goal_test(kb):
        required = [expr('At(Spare, Axle)'), expr('At(Flat, Ground)')]
        for q in required:
            if kb.ask(q) is False:
                return False
        return True

    ##Actions
    #Remove
    precond_pos = [expr("At(obj, loc)")]
    precond_neg = []
    effect_add = [expr("At(obj, Ground)")]
    effect_rem = [expr("At(obj, loc)")]
    remove = Action(expr("Remove(obj, loc)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #PutOn
    precond_pos = [expr("Tire(t)"), expr("At(t, Ground)")]
    precond_neg = [expr("At(Flat, Axle)")]
    effect_add = [expr("At(t, Axle)")]
    effect_rem = [expr("At(t, Ground)")]
    put_on = Action(expr("PutOn(t, Axle)"), [precond_pos, precond_neg], [effect_add, effect_rem])

    #LeaveOvernight
    precond_pos = []
    precond_neg = []
    effect_add = []
    effect_rem = [expr("At(Spare, Ground)"), expr("At(Spare, Axle)"), expr("At(Spare, Trunk)"),
                  expr("At(Flat, Ground)"), expr("At(Flat, Axle)"), expr("At(Flat, Trunk)")]
    leave_overnight = Action(expr("LeaveOvernight"), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDLL(init, [remove, put_on, leave_overnight], goal_test)

def three_block_tower():
    init = [expr('On(A, Table)'),
            expr('On(B, Table)'),
            expr('On(C, A)'),
            expr('Block(A)'),
            expr('Block(B)'),
            expr('Block(C)'),
            expr('Clear(B)'),
            expr('Clear(C)')]

    def goal_test(kb):
        required = [expr('On(A, B)'), expr('On(B, C)')]
        for q in required:
            if kb.ask(q) is False:
                return False
        return True

    ## Actions
    #  Move
    precond_pos = [expr('On(b, x)'), expr('Clear(b)'), expr('Clear(y)'), expr('Block(b)'), expr('Block(y)')]
    precond_neg = []
    effect_add = [expr('On(b, y)'), expr('Clear(x)')]
    effect_rem = [expr('On(b, x)'), expr('Clear(y)')]
    move = Action(expr('Move(b, x, y)'), [precond_pos, precond_neg], [effect_add, effect_rem])
    
    #  MoveToTable
    precond_pos = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    precond_neg = []
    effect_add = [expr('On(b, Table)'), expr('Clear(x)')]
    effect_rem = [expr('On(b, x)')]
    moveToTable = Action(expr('MoveToTable(b, x)'), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDLL(init, [move, moveToTable], goal_test)

def have_cake_and_eat_cake_too():
    init = [expr('Have(Cake)')]

    def goal_test(kb):
        required = [expr('Have(Cake)'), expr('Eaten(Cake)')]
        for q in required:
            if kb.ask(q) is False:
                return False
        return True

    ##Actions
    # Eat cake
    precond_pos = [expr('Have(Cake)')]
    precond_neg = []
    effect_add = [expr('Eaten(Cake)')]
    effect_rem = [expr('Have(Cake)')]
    eat_cake = Action(expr('Eat(Cake)'), [precond_pos, precond_neg], [effect_add, effect_rem])

    #Bake Cake
    precond_pos = []
    precond_neg = [expr('Have(Cake)')]
    effect_add = [expr('Have(Cake)')]
    effect_rem = []
    bake_cake = Action(expr('Bake(Cake)'), [precond_pos, precond_neg], [effect_add, effect_rem])

    return PDLL(init, [eat_cake, bake_cake], goal_test)

class Level():
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, poskb, negkb):
        self.poskb = poskb
        #Current state
        self.current_state_pos = poskb.clauses
        self.current_state_neg = negkb.clauses
        #Current action to current state link
        self.current_action_links_pos = {}
        self.current_action_links_neg = {}
        #Current state to action link      
        self.current_state_links_pos = {}
        self.current_state_links_neg = {}
        #Current action to next state link
        self.next_action_links = {}
        #Next state to current action link
        self.next_state_links_pos = {}
        self.next_state_links_neg = {}
        self.mutex = []


    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()


    def find_mutex(self):
        #Inconsistent effects
        for poseff in self.next_state_links_pos:
            #negeff = Expr('not'+poseff.op, poseff.args)
            negeff = poseff
            if negeff in self.next_state_links_neg:
                for a in self.next_state_links_pos[poseff]:
                    for b in self.next_state_links_neg[negeff]:
                        if set([a,b]) not in self.mutex:
                            self.mutex.append(set([a,b]))

        #Interference
        for posprecond in self.current_state_links_pos:
            #negeff = Expr('not'+posprecond.op, posprecond.args)
            negeff = posprecond
            if negeff in self.next_state_links_neg:
                for a in self.current_state_links_pos[posprecond]:
                    for b in self.next_state_links_neg[negeff]:
                        if set([a,b]) not in self.mutex:
                            self.mutex.append(set([a,b]))

        for negprecond in self.current_state_links_neg:
            #poseff = Expr(negprecond.op[3:], negprecond.args)
            poseff = negprecond
            if poseff in self.next_state_links_pos:
                for a in self.next_state_links_pos[poseff]:
                    for b in self.current_state_links_neg[negprecond]:
                        if set([a,b]) not in self.mutex:
                            self.mutex.append(set([a,b]))

        #Competing needs
        for posprecond in self.current_state_links_pos:
            #negprecond = Expr('not'+posprecond.op, posprecond.args)
            negprecond = posprecond
            if negprecond in self.current_state_links_neg:
                for a in self.current_state_links_pos[posprecond]:
                    for b in self.current_state_links_neg[negprecond]:
                        if set([a,b]) not in self.mutex:
                            self.mutex.append(set([a,b]))

        #Inconsistent support
        state_mutex = []
        for pair in self.mutex:
            next_state_0 = self.next_action_links[list(pair)[0]]
            if len(pair) == 2:
                next_state_1 = self.next_action_links[list(pair)[1]]
            else:
                next_state_1 = self.next_action_links[list(pair)[0]]
            if (len(next_state_0) == 1) and (len(next_state_1) == 1):
                state_mutex.append(set([next_state_0[0], next_state_1[0]]))

        self.mutex = self.mutex+state_mutex


    def build(self, actions, objects):

        #Add persistence actions for positive states
        for clause in self.current_state_pos:
            self.current_action_links_pos[Expr('Persistence', clause)] = [clause]
            self.next_action_links[Expr('Persistence', clause)] = [clause]
            self.current_state_links_pos[clause] = [Expr('Persistence', clause)]
            self.next_state_links_pos[clause] = [Expr('Persistence', clause)]

        #Add persistence actions for negative states
        for clause in self.current_state_neg:
            self.current_action_links_neg[Expr('Persistence', Expr('not'+clause.op, clause.args))] = [clause]
            self.next_action_links[Expr('Persistence', Expr('not'+clause.op, clause.args))] = [clause]
            self.current_state_links_neg[clause] = [Expr('Persistence', Expr('not'+clause.op, clause.args))]
            self.next_state_links_neg[clause] = [Expr('Persistence', Expr('not'+clause.op, clause.args))]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.poskb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower():
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links_pos[new_action] = []
                    self.current_action_links_neg[new_action] = []

                    for clause in a.precond_pos:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links_pos[new_action].append(new_clause)
                        if new_clause in self.current_state_links_pos:
                            self.current_state_links_pos[new_clause].append(new_action)
                        else:
                            self.current_state_links_pos[new_clause] = [new_action]

                    for clause in a.precond_neg:
                        new_clause = a.substitute(clause, arg)
                        #new_clause = Expr('not'+new_clause.op, new_clause.arg)
                        self.current_action_links_neg[new_action].append(new_clause)
                        if new_clause in self.current_state_links_neg:
                            self.current_state_links_neg[new_clause].append(new_action)
                        else:
                            self.current_state_links_neg[new_clause] = [new_action]

                    self.next_action_links[new_action] = []
                    for clause in a.effect_add:
                        new_clause = a.substitute(clause, arg)
                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links_pos:
                            self.next_state_links_pos[new_clause].append(new_action)
                        else:
                            self.next_state_links_pos[new_clause] = [new_action]

                    for clause in a.effect_rem:
                        new_clause = a.substitute(clause, arg)
                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links_neg:
                            self.next_state_links_neg[new_clause].append(new_action)
                        else:
                            self.next_state_links_neg[new_clause] = [new_action]


    def perform_actions(self):
        new_kb_pos, new_kb_neg = FolKB(list(set(self.next_state_links_pos.keys()))), FolKB(list(set(self.next_state_links_neg.keys())))
        return Level(new_kb_pos, new_kb_neg)


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, pdll, negkb):
        self.pdll = pdll
        self.levels = [Level(pdll.kb, negkb)]
        self.objects = set(arg for clause in pdll.kb.clauses + negkb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def expand_graph(self):
        last_level = self.levels[-1]
        last_level(self.pdll.actions, self.objects)
        self.levels.append(last_level.perform_actions())

    def non_mutex_goals(self, goals, index):
        goal_perm = itertools.combinations(goals, 2)
        for g in goal_perm:
            if set(g) in self.levels[index].mutex:
                return False
        return True


class GraphPlan:
    """
    Class for formulation GraphPlan algorithm
    Constructs a graph of state and action space
    Returns solution for the planning problem
    """

    def __init__(self, pdll, negkb):
        self.graph = Graph(pdll, negkb)
        self.nogoods = []
        self.solution = []

    def check_leveloff(self):
        if (set(self.graph.levels[-1].current_state_pos) == set(self.graph.levels[-2].current_state_pos)) and (set(lf.graph.levels[-1].current_state_neg) == set(self.graph.levels[-2].current_state_neg)):
            return True

    def extract_solution(self, goals_pos, goals_neg, index):
        level = self.graph.levels[index]
        if not self.graph.non_mutex_goals(goals_pos+goals_neg, index):
            self.nogoods.append((level, goals_pos, goals_neg))
            return

        level = self.graph.levels[index-1]

        #Create all combinations of actions that satisfy the goal
        actions = []
        for goal in goals_pos:
            actions.append(level.next_state_links_pos[goal])

        for goal in goals_neg:
            actions.append(level.next_state_links_neg[goal])

        all_actions = list(itertools.product(*actions))

        #Filter out the action combinations which contain mutexes
        non_mutex_actions = []
        for action_tuple in all_actions:
            action_pairs = itertools.combinations(list(set(action_tuple)), 2)
            non_mutex_actions.append(list(set(action_tuple)))
            for pair in action_pairs:
                if set(pair) in level.mutex:
                    non_mutex_actions.pop(-1)
                    break

        #Recursion
        for action_list in non_mutex_actions:
            if [action_list, index] not in self.solution:
                self.solution.append([action_list, index])

                new_goals_pos = []
                new_goals_neg = []
                for act in set(action_list):
                    if act in level.current_action_links_pos:
                        new_goals_pos = new_goals_pos + level.current_action_links_pos[act]

                for act in set(action_list):
                    if act in level.current_action_links_neg:
                        new_goals_neg = new_goals_neg + level.current_action_links_neg[act]

                if abs(index)+1 == len(self.graph.levels):
                    return
                elif (level, new_goals_pos, new_goals_neg) in self.nogoods:
                    return
                else:
                    self.extract_solution(new_goals_pos, new_goals_neg, index-1)

        #Level-Order multiple solutions
        solution = []
        for item in self.solution:
            if item[1] == -1:
                solution.append([])
                solution[-1].append(item[0])
            else:
                solution[-1].append(item[0])

        for num, item in enumerate(solution):
            item.reverse()
            solution[num] = item

        return solution


def goal_test(kb, goals):
    for q in goals:
        if kb.ask(q) is False:
            return False
    return True


def spare_tire_graphplan():
    pdll = spare_tire()
    negkb = FolKB([expr('At(Flat, Trunk)')])
    graphplan = GraphPlan(pdll, negkb)
    ##Not sure
    goals_pos = [expr('At(Spare, Axle)'), expr('At(Flat, Ground)')]
    goals_neg = []

    while True:
        if goal_test(graphplan.graph.levels[-1].poskb, goals_pos) and graphplan.graph.non_mutex_goals(goals_pos+goals_neg, -1):
            solution = graphplan.extract_solution(goals_pos, goals_neg, -1)
            if solution:
                return solution
        graphplan.graph.expand_graph()
        if len(graphplan.graph.levels)>=2 and graphplan.check_leveloff():
            return None
