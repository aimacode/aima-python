"""Planning (Chapters 11)
"""

import copy
import itertools
from search import Node
from utils import Expr, expr, first
from logic import FolKB, conjuncts, unify
from collections import deque
from functools import reduce as _reduce

# ____________________________________________________________
# 11.1. Deﬁnition of Classical Planning


class PlanningProblem:
    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, init, goals, actions):
        self.init = self.convert(init)
        self.goals = self.convert(goals)
        self.actions = actions

    def convert(self, clauses):
        """Converts strings into exprs"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.init for goal in self.goals)

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
        if not list_action.check_precond(self.init, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.init = list_action(self.init, args).clauses


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PlanningProblem.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negative literals.
    Negative preconditions and effects are defined by adding a 'Not' before the name of the clause
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond)
        self.effect = self.convert(effect)

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, Expr(self.name, *self.args))

    def convert(self, clauses):
        """Converts strings into Exprs"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propositional symbol"""

        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""

        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's knowledge base"""

        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            kb.tell(self.substitute(clause, args))
            if clause.op[:3] == 'Not':
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:    
                    kb.retract(self.substitute(new_clause, args))

        return kb


def goal_test(goals, state):
    """Generic goal testing helper function"""

    if isinstance(state, list):
        kb = FolKB(state)
    else:
        kb = state
    return all(kb.ask(q) is not False for q in goals)

# 11.1.1 Example domain: Air cargo transport


def air_cargo():
    """
    [Figure 11.1] AIR-CARGO-PROBLEM

    An air-cargo shipment problem for delivering cargo to different locations,
    given the starting location and airplanes.

    Example:
    >>> from planning import *
    >>> ac = air_cargo()
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Load(C2, P2, JFK)'))
    >>> ac.act(expr('Load(C1, P1, SFO)'))
    >>> ac.act(expr('Fly(P1, SFO, JFK)'))
    >>> ac.act(expr('Fly(P2, JFK, SFO)'))
    >>> ac.act(expr('Unload(C2, P2, SFO)'))
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Unload(C1, P1, JFK)'))
    >>> ac.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK) & Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)', 
                goals='At(C1, JFK) & At(C2, SFO)',
                actions=[Action('Load(c, p, a)', 
                                precond='At(c, a) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)',
                                effect='In(c, p) & ~At(c, a)'),
                         Action('Unload(c, p, a)',
                                precond='In(c, p) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)',
                                effect='At(c, a) & ~In(c, p)'),
                         Action('Fly(p, f, to)',
                                precond='At(p, f) & Plane(p) & Airport(f) & Airport(to)',
                                effect='At(p, to) & ~At(p, f)')])

# 11.1.2 Example domain: The spare tire problem


def spare_tire():
    """[Figure 11.2] SPARE-TIRE-PROBLEM

    A problem involving changing the flat tire of a car
    with a spare tire from the trunk.

    Example:
    >>> from planning import *
    >>> st = spare_tire()
    >>> st.goal_test()
    False
    >>> st.act(expr('Remove(Spare, Trunk)'))
    >>> st.act(expr('Remove(Flat, Axle)'))
    >>> st.goal_test()
    False
    >>> st.act(expr('PutOn(Spare, Axle)'))
    >>> st.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='Tire(Flat) & Tire(Spare) & At(Flat, Axle) & At(Spare, Trunk)',
                goals='At(Spare, Axle) & At(Flat, Ground)',
                actions=[Action('Remove(obj, loc)',
                                precond='At(obj, loc)',
                                effect='At(obj, Ground) & ~At(obj, loc)'),
                         Action('PutOn(t, Axle)',
                                precond='Tire(t) & At(t, Ground) & ~At(Flat, Axle)',
                                effect='At(t, Axle) & ~At(t, Ground)'),
                         Action('LeaveOvernight',
                                precond='',
                                effect='~At(Spare, Ground) & ~At(Spare, Axle) & ~At(Spare, Trunk) & \
                                        ~At(Flat, Ground) & ~At(Flat, Axle) & ~At(Flat, Trunk)')])


def three_block_tower():
    """
    [Figure 11.3] THREE-BLOCK-TOWER

    A blocks-world problem of stacking three blocks in a certain configuration,
    also known as the Sussman Anomaly.

    Example:
    >>> from planning import *
    >>> tbt = three_block_tower()
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('MoveToTable(C, A)'))
    >>> tbt.act(expr('Move(B, Table, C)'))
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('Move(A, Table, B)'))
    >>> tbt.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='On(A, Table) & On(B, Table) & On(C, A) & Block(A) & Block(B) & Block(C) & Clear(B) & Clear(C)',
                goals='On(A, B) & On(B, C)',
                actions=[Action('Move(b, x, y)',
                                precond='On(b, x) & Clear(b) & Clear(y) & Block(b) & Block(y)',
                                effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)'),
                         Action('MoveToTable(b, x)',
                                precond='On(b, x) & Clear(b) & Block(b)',
                                effect='On(b, Table) & Clear(x) & ~On(b, x)')])

# 11.1.3 Example domain: The blocks world


def simple_blocks_world():
    """
    SIMPLE-BLOCKS-WORLD

    A simplified definition of the Sussman Anomaly problem.

    Example:
    >>> from planning import *
    >>> sbw = simple_blocks_world()
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('ToTable(A, B)'))
    >>> sbw.act(expr('FromTable(B, A)'))
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('FromTable(C, B)'))
    >>> sbw.goal_test()
    True
    >>>
    """

    return PlanningProblem(init='On(A, B) & Clear(A) & OnTable(B) & OnTable(C) & Clear(C)',
                goals='On(B, A) & On(C, B)',
                actions=[Action('ToTable(x, y)',
                                precond='On(x, y) & Clear(x)',
                                effect='~On(x, y) & Clear(y) & OnTable(x)'),
                         Action('FromTable(y, x)',
                                precond='OnTable(y) & Clear(y) & Clear(x)',
                                effect='~OnTable(y) & ~Clear(x) & On(y, x)')])


class Level:
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, kb):
        """Initializes variables to hold state and action details of a level"""

        self.kb = kb
        # current state
        self.current_state = kb.clauses
        # current action to state link
        self.current_action_links = {}
        # current state to action link
        self.current_state_links = {}
        # current action to next state link
        self.next_action_links = {}
        # next state to current action link
        self.next_state_links = {}
        # mutually exclusive actions
        self.mutex = []

    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()

    def separate(self, e):
        """Separates an iterable of elements into positive and negative parts"""

        positive = []
        negative = []
        for clause in e:
            if clause.op[:3] == 'Not':
                negative.append(clause)
            else:
                positive.append(clause)
        return positive, negative

    def find_mutex(self):
        """Finds mutually exclusive actions"""

        # Inconsistent effects
        pos_nsl, neg_nsl = self.separate(self.next_state_links)

        for negeff in neg_nsl:
            new_negeff = Expr(negeff.op[3:], *negeff.args)
            for poseff in pos_nsl:
                if new_negeff == poseff:
                    for a in self.next_state_links[poseff]:
                        for b in self.next_state_links[negeff]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Interference will be calculated with the last step
        pos_csl, neg_csl = self.separate(self.current_state_links)

        # Competing needs
        for posprecond in pos_csl:
            for negprecond in neg_csl:
                new_negprecond = Expr(negprecond.op[3:], *negprecond.args)
                if new_negprecond == posprecond:
                    for a in self.current_state_links[posprecond]:
                        for b in self.current_state_links[negprecond]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Inconsistent support
        state_mutex = []
        for pair in self.mutex:
            next_state_0 = self.next_action_links[list(pair)[0]]
            if len(pair) == 2:
                next_state_1 = self.next_action_links[list(pair)[1]]
            else:
                next_state_1 = self.next_action_links[list(pair)[0]]
            if (len(next_state_0) == 1) and (len(next_state_1) == 1):
                state_mutex.append({next_state_0[0], next_state_1[0]})
        
        self.mutex = self.mutex + state_mutex

    def build(self, actions, objects):
        """Populates the lists and dictionaries containing the state action dependencies"""

        for clause in self.current_state:
            p_expr = Expr('P' + clause.op, *clause.args)
            self.current_action_links[p_expr] = [clause]
            self.next_action_links[p_expr] = [clause]
            self.current_state_links[clause] = [p_expr]
            self.next_state_links[clause] = [p_expr]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.kb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower():
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links[new_action] = []

                    for clause in a.precond:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links[new_action].append(new_clause)
                        if new_clause in self.current_state_links:
                            self.current_state_links[new_clause].append(new_action)
                        else:
                            self.current_state_links[new_clause] = [new_action]
                   
                    self.next_action_links[new_action] = []
                    for clause in a.effect:
                        new_clause = a.substitute(clause, arg)

                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links:
                            self.next_state_links[new_clause].append(new_action)
                        else:
                            self.next_state_links[new_clause] = [new_action]

    def perform_actions(self):
        """Performs the necessary actions and returns a new Level"""

        new_kb = FolKB(list(set(self.next_state_links.keys())))
        return Level(new_kb)

# ___________________________________________________________
# 11.4 Hierarchical Planning
# 11.4.1 High-level actions


class HLA(Action):
    """
    Define Actions for the real-world (that may be refined further), and satisfy resource
    constraints.
    """
    unique_group = 1

    def __init__(self, action, precond=None, effect=None, duration=0,
                 consume=None, use=None):
        """
        As opposed to actions, to define HLA, we have added constraints.
        duration holds the amount of time required to execute the task
        consumes holds a dictionary representing the resources the task consumes
        uses holds a dictionary representing the resources the task uses
        """
        precond = precond or [None]
        effect = effect or [None]
        super().__init__(action, precond, effect)
        self.duration = duration
        self.consumes = consume or {}
        self.uses = use or {}
        self.completed = False
        # self.priority = -1 #  must be assigned in relation to other HLAs
        # self.job_group = -1 #  must be assigned in relation to other HLAs

    def do_action(self, job_order, available_resources, kb, args):
        """
        An HLA based version of act - along with knowledge base updation, it handles
        resource checks, and ensures the actions are executed in the correct order.
        """
        # print(self.name)
        if not self.has_usable_resource(available_resources):
            raise Exception('Not enough usable resources to execute {}'.format(self.name))
        if not self.has_consumable_resource(available_resources):
            raise Exception('Not enough consumable resources to execute {}'.format(self.name))
        if not self.inorder(job_order):
            raise Exception("Can't execute {} - execute prerequisite actions first".
                            format(self.name))
        kb = super().act(kb, args)  # update knowledge base
        for resource in self.consumes:  # remove consumed resources
            available_resources[resource] -= self.consumes[resource]
        self.completed = True  # set the task status to complete
        return kb

    def has_consumable_resource(self, available_resources):
        """
        Ensure there are enough consumable resources for this action to execute.
        """
        for resource in self.consumes:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.consumes[resource]:
                return False
        return True

    def has_usable_resource(self, available_resources):
        """
        Ensure there are enough usable resources for this action to execute.
        """
        for resource in self.uses:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.uses[resource]:
                return False
        return True

    def inorder(self, job_order):
        """
        Ensure that all the jobs that had to be executed before the current one have been
        successfully executed.
        """
        for jobs in job_order:
            if self in jobs:
                for job in jobs:
                    if job is self:
                        return True
                    if not job.completed:
                        return False
        return True

# 11.4.2 Searching for primitive solutions


class Problem(PlanningProblem):
    """
    Define real-world problems by aggregating resources as numerical quantities instead of
    named entities.

    This class is identical to PDLL, except that it overloads the act function to handle
    resource and ordering conditions imposed by HLA as opposed to Action.
    """
    def __init__(self, init, goals, actions, jobs=None, resources=None):
        super().__init__(init, goals, actions)
        self.jobs = jobs
        self.resources = resources or {}

    def act(self, action):
        """
        Performs the HLA given as argument.

        Note that this is different from the superclass action - where the parameter was an
        Expression. For real world problems, an Expr object isn't enough to capture all the
        detail required for executing the action - resources, preconditions, etc need to be
        checked for too.
        """
        args = action.args
        list_action = first(a for a in self.actions if a.name == action.name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action.name))
        self.init = list_action.do_action(self.jobs, self.resources, self.init, args).clauses

    def refinements(hla, state, library):  # refinements may be (multiple) HLA themselves ...
        """
        state is a Problem, containing the current state kb
        library is a dictionary containing details for every possible refinement. eg:
        {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
            ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
            ],
        # empty refinements indicate a primitive action
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
            ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
            ]
        }
        """
        e = Expr(hla.name, hla.args)
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
        for i in indices:
            actions = []
            for j in range(len(library['steps'][i])):
                # find the index of the step [j]  of the HLA 
                index_step = [k for k,x in enumerate(library['HLA']) if x == library['steps'][i][j]][0]
                precond = library['precond'][index_step][0] # preconditions of step [j]
                effect = library['effect'][index_step][0] # effect of step [j]
                actions.append(HLA(library['steps'][i][j], precond, effect))
            yield actions

    def hierarchical_search(problem, hierarchy):
        """
        [Figure 11.8] 'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world problem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details)
        """
        act = Node(problem.init, None, [problem.actions[0]])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            (hla, index) = Problem.find_hla(plan, hierarchy) # finds the first non primitive hla in plan actions
            prefix = plan.action[:index]
            outcome = Problem(Problem.result(problem.init, prefix), problem.goals , problem.actions )
            suffix = plan.action[index+1:]
            if not hla: # hla is None and plan is primitive
                if outcome.goal_test():
                    return plan.action
            else:
                for sequence in Problem.refinements(hla, outcome, hierarchy): # find refinements
                    frontier.append(Node(outcome.init, plan, prefix + sequence+ suffix))

    def result(state, actions):
        """The outcome of applying an action to the current problem"""
        for a in actions: 
            if a.check_precond(state, a.args):
                state = a(state, a.args).clauses
        return state

    def angelic_search(problem, hierarchy, initialPlan):
        """
        [Figure 11.11] A hierarchical planning algorithm that uses angelic semantics to identify and
        commit to high-level plans that work while avoiding high-level plans that don’t.
        The predicate MAKING-PROGRESS checks to make sure that we aren’t stuck in an infinite regression
        of refinements.
        At top level, call ANGELIC -SEARCH with [Act ] as the initialPlan .

        initialPlan contains a sequence of HLA's with angelic semantics

        The possible effects of an angelic HLA in initialPlan are : 
        ~ : effect remove
        $+: effect possibly add
        $-: effect possibly remove
        $$: possibly add or remove
        """
        frontier = deque(initialPlan)
        while True: 
            if not frontier:
                return None
            plan = frontier.popleft() # sequence of HLA/Angelic HLA's 
            opt_reachable_set = Problem.reach_opt(problem.init, plan)
            pes_reachable_set = Problem.reach_pes(problem.init, plan)
            if problem.intersects_goal(opt_reachable_set): 
                if Problem.is_primitive( plan, hierarchy ): 
                    return ([x for x in plan.action])
                guaranteed = problem.intersects_goal(pes_reachable_set) 
                if guaranteed and Problem.making_progress(plan, initialPlan):
                    final_state = guaranteed[0] # any element of guaranteed 
                    return Problem.decompose(hierarchy, problem, plan, final_state, pes_reachable_set)
                hla, index = Problem.find_hla(plan, hierarchy) # there should be at least one HLA/Angelic_HLA, otherwise plan would be primitive.
                prefix = plan.action[:index]
                suffix = plan.action[index+1:]
                outcome = Problem(Problem.result(problem.init, prefix), problem.goals , problem.actions )
                for sequence in Problem.refinements(hla, outcome, hierarchy): # find refinements
                    frontier.append(Angelic_Node(outcome.init, plan, prefix + sequence+ suffix, prefix+sequence+suffix))

    def intersects_goal(problem, reachable_set):
        """
        Find the intersection of the reachable states and the goal
        """
        return [y for x in list(reachable_set.keys()) for y in reachable_set[x] if all(goal in y for goal in problem.goals)] 

    def is_primitive(plan,  library):
        """
        checks if the hla is primitive action 
        """
        for hla in plan.action: 
            indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
            for i in indices:
                if library["steps"][i]: 
                    return False
        return True

    def reach_opt(init, plan): 
        """
        Finds the optimistic reachable set of the sequence of actions in plan 
        """
        reachable_set = {0: [init]}
        optimistic_description = plan.action #list of angelic actions with optimistic description
        return Problem.find_reachable_set(reachable_set, optimistic_description)

    def reach_pes(init, plan): 
        """ 
        Finds the pessimistic reachable set of the sequence of actions in plan
        """
        reachable_set = {0: [init]}
        pessimistic_description = plan.action_pes # list of angelic actions with pessimistic description
        return Problem.find_reachable_set(reachable_set, pessimistic_description)

    def find_reachable_set(reachable_set, action_description):
        """
        Finds the reachable states of the action_description when applied in each state of reachable set.
        """
        for i in range(len(action_description)):
            reachable_set[i+1]=[]
            if type(action_description[i]) is Angelic_HLA:
                possible_actions = action_description[i].angelic_action()
            else: 
                possible_actions = action_description
            for action in possible_actions:
                for state in reachable_set[i]:
                    if action.check_precond(state , action.args) :
                        if action.effect[0] :
                            new_state = action(state, action.args).clauses
                            reachable_set[i+1].append(new_state)
                        else: 
                            reachable_set[i+1].append(state)
        return reachable_set

    def find_hla(plan, hierarchy):
        """
        Finds the the first HLA action in plan.action, which is not primitive
        and its corresponding index in plan.action
        """
        hla = None
        index = len(plan.action)
        for i in range(len(plan.action)): # find the first HLA in plan, that is not primitive
            if not Problem.is_primitive(Node(plan.state, plan.parent, [plan.action[i]]), hierarchy):
                hla = plan.action[i] 
                index = i
                break
        return hla, index

    def making_progress(plan, initialPlan):
        """ 
        Prevents from infinite regression of refinements  

        (infinite regression of refinements happens when the algorithm finds a plan that 
        its pessimistic reachable set intersects the goal inside a call to decompose on the same plan, in the same circumstances)  
        """
        for i in range(len(initialPlan)):
            if (plan == initialPlan[i]):
                return False
        return True 

    def decompose(hierarchy, s_0, plan, s_f, reachable_set):
        solution = [] 
        i = max(reachable_set.keys())
        while plan.action_pes: 
            action = plan.action_pes.pop()
            if (i==0): 
                return solution
            s_i = Problem.find_previous_state(s_f, reachable_set,i, action) 
            problem = Problem(s_i, s_f , plan.action)
            angelic_call = Problem.angelic_search(problem, hierarchy, [Angelic_Node(s_i, Node(None), [action],[action])])
            if angelic_call:
                for x in angelic_call: 
                    solution.insert(0,x)
            else: 
                return None
            s_f = s_i
            i-=1
        return solution

    def find_previous_state(s_f, reachable_set, i, action):
        """
        Given a final state s_f and an action finds a state s_i in reachable_set 
        such that when action is applied to state s_i returns s_f.  
        """
        s_i = reachable_set[i-1][0]
        for state in reachable_set[i-1]:
            if s_f in [x for x in Problem.reach_pes(state, Angelic_Node(state, None, [action],[action]))[1]]:
                s_i =state
                break
        return s_i

# ___________________________________________________________________
# 11.6 Time, Schedule, and Resources


def job_shop_problem():
    """
    [Figure 11.13] JOB-SHOP-PROBLEM

    A job-shop scheduling problem for assembling two cars,
    with resource and ordering constraints.

    Example:
    >>> from planning import *
    >>> p = job_shop_problem()
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[1][0])
    >>> p.act(p.jobs[1][1])
    >>> p.act(p.jobs[1][2])
    >>> p.act(p.jobs[0][0])
    >>> p.act(p.jobs[0][1])
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[0][2])
    >>> p.goal_test()
    True
    >>>
    """
    resources = {'EngineHoists': 1, 'WheelStations': 2, 'Inspectors': 2, 'LugNuts': 500}

    add_engine1 = HLA('AddEngine1', precond='~Has(C1, E1)', effect='Has(C1, E1)', duration=30, use={'EngineHoists': 1})
    add_engine2 = HLA('AddEngine2', precond='~Has(C2, E2)', effect='Has(C2, E2)', duration=60, use={'EngineHoists': 1})
    add_wheels1 = HLA('AddWheels1', precond='~Has(C1, W1)', effect='Has(C1, W1)', duration=30, use={'WheelStations': 1}, consume={'LugNuts': 20})
    add_wheels2 = HLA('AddWheels2', precond='~Has(C2, W2)', effect='Has(C2, W2)', duration=15, use={'WheelStations': 1}, consume={'LugNuts': 20})
    inspect1 = HLA('Inspect1', precond='~Inspected(C1)', effect='Inspected(C1)', duration=10, use={'Inspectors': 1})
    inspect2 = HLA('Inspect2', precond='~Inspected(C2)', effect='Inspected(C2)', duration=10, use={'Inspectors': 1})

    actions = [add_engine1, add_engine2, add_wheels1, add_wheels2, inspect1, inspect2]

    job_group1 = [add_engine1, add_wheels1, inspect1]
    job_group2 = [add_engine2, add_wheels2, inspect2]

    return Problem(init='Car(C1) & Car(C2) & Wheels(W1) & Wheels(W2) & Engine(E2) & Engine(E2) & ~Has(C1, E1) & ~Has(C2, E2) & ~Has(C1, W1) & ~Has(C2, W2) & ~Inspected(C1) & ~Inspected(C2)',
                   goals='Has(C1, W1) & Has(C1, E1) & Inspected(C1) & Has(C2, W2) & Has(C2, E2) & Inspected(C2)',
                   actions=actions,
                   jobs=[job_group1, job_group2],
                   resources=resources)


def go_to_sfo():
    """Go to SFO Problem"""

    go_home_sfo1 = HLA('Go(Home, SFO)', precond='At(Home) & Have(Car)', effect='At(SFO) & ~At(Home)')
    go_home_sfo2 = HLA('Go(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
    drive_home_sfoltp = HLA('Drive(Home, SFOLongTermParking)', precond='At(Home) & Have(Car)', effect='At(SFOLongTermParking) & ~At(Home)')
    shuttle_sfoltp_sfo = HLA('Shuttle(SFOLongTermParking, SFO)', precond='At(SFOLongTermParking)', effect='At(SFO) & ~At(SFOLongTermParking)')
    taxi_home_sfo = HLA('Taxi(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')

    actions = [go_home_sfo1, go_home_sfo2, drive_home_sfoltp, shuttle_sfoltp_sfo, taxi_home_sfo]

    library = {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
        ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
        ],
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
        ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
        ]
    }

    return Problem(init='At(Home)', goals='At(SFO)', actions=actions), library


class Angelic_HLA(HLA):
    """
    Define Actions for the real-world (that may be refined further), under angelic semantics
    """
    
    def __init__(self, action, precond , effect, duration =0, consume = None, use = None):
        super().__init__(action, precond, effect, duration, consume, use)

    def convert(self, clauses):
        """
        Converts strings into Exprs
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable ) 
        and furthermore can have following effects on the variables: 
            Possibly add variable    ( $+ )
            Possibly remove variable ( $- )
            Possibly add or remove a variable ( $$ )

        Overrides HLA.convert function
        """ 
        lib = {'~': 'Not', 
                '$+': 'PosYes',
               '$-': 'PosNot',
               '$$' : 'PosYesNot'}

        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                for ch in lib.keys():
                    if clauses[i].op == ch:
                        clauses[i] = expr( lib[ch] + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            for ch in lib.keys():
                clauses = clauses.replace(ch, lib[ch])
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def angelic_action(self):
        """
        Converts a high level action (HLA) with angelic semantics into all of its corresponding high level actions (HLA). 
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable) 
        and furthermore can have following effects for each variable: 

            Possibly add variable ( $+: 'PosYes' )        --> corresponds to two HLAs: 
                                                                HLA_1: add variable  
                                                                HLA_2: leave variable unchanged

            Possibly remove variable ( $-: 'PosNot' )     --> corresponds to two HLAs:
                                                                HLA_1: remove variable
                                                                HLA_2: leave variable unchanged

            Possibly add / remove a variable ( $$: 'PosYesNot' )  --> corresponds to three HLAs: 
                                                                        HLA_1: add variable
                                                                        HLA_2: remove variable
                                                                        HLA_3: leave variable unchanged 


            example: the angelic action with effects possibly add A and possibly add or remove B corresponds to the following 6 effects of HLAs:
            
            
            '$+A & $$B':    HLA_1: 'A & B'   (add A and add B)
                            HLA_2: 'A & ~B'  (add A and remove B)
                            HLA_3: 'A'       (add A)
                            HLA_4: 'B'       (add B)
                            HLA_5: '~B'      (remove B)
                            HLA_6: ' '       (no effect)  

        """

        effects=[[]]
        for clause in self.effect:
            (n,w) = Angelic_HLA.compute_parameters(clause, effects)
            effects = effects*n # create n copies of effects 
            it=range(1)
            if len(effects)!=0:
                # split effects into n sublists (seperate n copies created in compute_parameters)
                it = range(len(effects)//n)
            for i in it:
                if effects[i]:
                    if clause.args: 
                        effects[i] = expr(str(effects[i]) + '&' + str(Expr(clause.op[w:],clause.args[0]))) # make changes in the ith part of effects
                        if n==3:
                            effects[i+len(effects)//3]= expr(str(effects[i+len(effects)//3]) + '&' + str(Expr(clause.op[6:],clause.args[0])))
                    else: 
                        effects[i] = expr(str(effects[i]) + '&' + str(expr(clause.op[w:]))) # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = expr(str(effects[i+len(effects)//3]) + '&' + str(expr(clause.op[6:])))

                else: 
                    if clause.args: 
                        effects[i] = Expr(clause.op[w:], clause.args[0]) # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = Expr(clause.op[6:], clause.args[0])

                    else: 
                        effects[i] = expr(clause.op[w:])  # make changes in the ith part of effects
                        if n==3: 
                            effects[i+len(effects)//3] = expr(clause.op[6:])
            #print('effects',  effects)

        return [ HLA(Expr(self.name, self.args), self.precond, effects[i] ) for i in range(len(effects)) ]

    def compute_parameters(clause, effects):
        """ 
        computes n,w 
        
        n = number of HLA effects that the anelic HLA corresponds to
        w = length of representation of angelic HLA effect 

                    n = 1, if effect is add
                    n = 1, if effect is remove
                    n = 2, if effect is possibly add
                    n = 2, if effect is possibly remove
                    n = 3, if effect is possibly add or remove

        """
        if clause.op[:9] == 'PosYesNot':
            # possibly add/remove variable: three possible effects for the variable 
            n=3
            w=9
        elif clause.op[:6] == 'PosYes': # possibly add variable: two possible effects for the variable
            n=2
            w=6
        elif clause.op[:6] == 'PosNot': # possibly remove variable: two possible effects for the variable
            n=2
            w=3 # We want to keep 'Not' from 'PosNot' when adding action
        else:   # variable or ~variable 
            n=1
            w=0
        return (n,w)


class Angelic_Node(Node):
    """ 
    Extends the class Node. 
    self.action:     contains the optimistic description of an angelic HLA
    self.action_pes: contains the pessimistic description of an angelic HLA
    """

    def __init__(self, state, parent=None, action_opt=None, action_pes=None,  path_cost=0):
        super().__init__(state, parent, action_opt , path_cost)
        self.action_pes = action_pes 
