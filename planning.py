"""Planning (Chapters 10-11)
"""
import os
import copy
import itertools
from search import Node, astar_search
from collections import deque
from logic import fol_bc_and, FolKB, conjuncts
from utils import expr, Expr, partition, first
from pddl_parse import DomainParser, ProblemParser, build_expr_string


class PDDL:
    """
    Planning Domain Definition Language (PDDL) used to define a search problem.
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
            clauses = clauses

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
    Use this to describe actions in PDDL.
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


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, pddl):
        self.pddl = pddl
        self.kb = FolKB(pddl.init)
        self.levels = [Level(self.kb)]
        self.objects = set(arg for clause in self.kb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def expand_graph(self):
        """Expands the graph by a level"""

        last_level = self.levels[-1]
        last_level(self.pddl.actions, self.objects)
        self.levels.append(last_level.perform_actions())

    def non_mutex_goals(self, goals, index):
        """Checks whether the goals are mutually exclusive"""

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

    def __init__(self, pddl):
        self.graph = Graph(pddl)
        self.nogoods = []
        self.solution = []

    def check_leveloff(self):
        """Checks if the graph has levelled off"""

        check = (set(self.graph.levels[-1].current_state) == set(self.graph.levels[-2].current_state))

        if check:
            return True

    def extract_solution(self, goals, index):
        """Extracts the solution"""

        level = self.graph.levels[index]    
        if not self.graph.non_mutex_goals(goals, index):
            self.nogoods.append((level, goals))
            return

        level = self.graph.levels[index - 1]    

        # Create all combinations of actions that satisfy the goal    
        actions = []
        for goal in goals:
            actions.append(level.next_state_links[goal])    

        all_actions = list(itertools.product(*actions))    

        # Filter out non-mutex actions
        non_mutex_actions = []    
        for action_tuple in all_actions:
            action_pairs = itertools.combinations(list(set(action_tuple)), 2)        
            non_mutex_actions.append(list(set(action_tuple)))        
            for pair in action_pairs:            
                if set(pair) in level.mutex:
                    non_mutex_actions.pop(-1)
                    break
    

        # Recursion
        for action_list in non_mutex_actions:        
            if [action_list, index] not in self.solution:
                self.solution.append([action_list, index])

                new_goals = []
                for act in set(action_list):                
                    if act in level.current_action_links:
                        new_goals = new_goals + level.current_action_links[act]

                if abs(index) + 1 == len(self.graph.levels):
                    return
                elif (level, new_goals) in self.nogoods:
                    return
                else:
                    self.extract_solution(new_goals, index - 1)

        # Level-Order multiple solutions
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

    def goal_test(self, kb):
        return all(kb.ask(q) is not False for q in self.graph.pddl.goals)

    def execute(self):
        """Executes the GraphPlan algorithm for the given problem"""

        while True:
            self.graph.expand_graph()
            if (self.goal_test(self.graph.levels[-1].kb) and self.graph.non_mutex_goals(self.graph.pddl.goals, -1)):
                solution = self.extract_solution(self.graph.pddl.goals, -1)
                if solution:
                    return solution

            if len(self.graph.levels) >= 2 and self.check_leveloff():
                return None


def spare_tire():
    """Spare tire problem"""

    return PDDL(init='Tire(Flat) & Tire(Spare) & At(Flat, Axle) & At(Spare, Trunk)',
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
    """Sussman Anomaly problem"""

    return PDDL(init='On(A, Table) & On(B, Table) & On(C, A) & Block(A) & Block(B) & Block(C) & Clear(B) & Clear(C)',
                goals='On(A, B) & On(B, C)',
                actions=[Action('Move(b, x, y)',
                                precond='On(b, x) & Clear(b) & Clear(y) & Block(b) & Block(y)',
                                effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)'),
                         Action('MoveToTable(b, x)',
                                precond='On(b, x) & Clear(b) & Block(b)',
                                effect='On(b, Table) & Clear(x) & ~On(b, x)')])


def have_cake_and_eat_cake_too():
    """Cake problem"""

    return PDDL(init='Have(Cake)',
                goals='Have(Cake) & Eaten(Cake)',
                actions=[Action('Eat(Cake)',
                                precond='Have(Cake)',
                                effect='Eaten(Cake) & ~Have(Cake)'),
                         Action('Bake(Cake)',
                                precond='~Have(Cake)',
                                effect='Have(Cake)')])


def shopping_problem():
    """Shopping problem"""

    return PDDL(init='At(Home) & Sells(SM, Milk) & Sells(SM, Banana) & Sells(HW, Drill)',
                goals='Have(Milk) & Have(Banana) & Have(Drill)',
                actions=[Action('Buy(x, store)',
                                precond='At(store) & Sells(store, x)',
                                effect='Have(x)'),
                         Action('Go(x, y)',
                                precond='At(x)',
                                effect='At(y) & ~At(x)')])


def socks_and_shoes():
    """Socks and shoes problem"""

    return PDDL(init='',
                goals='RightShoeOn & LeftShoeOn',
                actions=[Action('RightShoe',
                                precond='RightSockOn',
                                effect='RightShoeOn'),
                        Action('RightSock',
                                precond='',
                                effect='RightSockOn'),
                        Action('LeftShoe',
                                precond='LeftSockOn',
                                effect='LeftShoeOn'),
                        Action('LeftSock',
                                precond='',
                                effect='LeftSockOn')])


def air_cargo():
    """Air cargo problem"""

    return PDDL(init='At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK) & Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)',
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


def spare_tire_graphplan():
    """Solves the spare tire problem using GraphPlan"""

    pddl = spare_tire()
    graphplan = GraphPlan(pddl)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    goals = expr('At(Spare, Axle), At(Flat, Ground)')

    while True:
        graphplan.graph.expand_graph()
        if (goal_test(graphplan.graph.levels[-1].kb, goals) and graphplan.graph.non_mutex_goals(goals, -1)):
            solution = graphplan.extract_solution(goals, -1)
            if solution:
                return solution
        
        if len(graphplan.graph.levels) >= 2 and graphplan.check_leveloff():
            return None


def have_cake_and_eat_cake_too_graphplan():
    """Solves the cake problem using GraphPlan"""

    pddl = have_cake_and_eat_cake_too()
    graphplan = GraphPlan(pddl)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    goals = expr('Have(Cake), Eaten(Cake)')

    while True:
        graphplan.graph.expand_graph()
        if (goal_test(graphplan.graph.levels[-1].kb, goals) and graphplan.graph.non_mutex_goals(goals, -1)):
            solution = graphplan.extract_solution(goals, -1)
            if solution:
                return [solution[1]]

        if len(graphplan.graph.levels) >= 2 and graphplan.check_leveloff():
            return None


def three_block_tower_graphplan():
    """Solves the Sussman Anomaly problem using GraphPlan"""

    pddl = three_block_tower()
    graphplan = GraphPlan(pddl)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    goals = expr('On(A, B), On(B, C)')

    while True:
        if (goal_test(graphplan.graph.levels[-1].kb, goals) and graphplan.graph.non_mutex_goals(goals, -1)):
            solution = graphplan.extract_solution(goals, -1)
            if solution:
                return solution

        graphplan.graph.expand_graph()
        if len(graphplan.graph.levels) >= 2 and graphplan.check_leveloff():
            return None


def air_cargo_graphplan():
    """Solves the air cargo problem using GraphPlan"""

    pddl = air_cargo()
    graphplan = GraphPlan(pddl)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    goals = expr('At(C1, JFK), At(C2, SFO)')

    while True:
        if (goal_test(graphplan.graph.levels[-1].kb, goals) and graphplan.graph.non_mutex_goals(goals, -1)):
            solution = graphplan.extract_solution(goals, -1)
            if solution:
                return solution

        graphplan.graph.expand_graph()
        if len(graphplan.graph.levels) >= 2 and graphplan.check_leveloff():
            return None


def socks_and_shoes_graphplan():
    pddl = socks_and_shoes()
    graphplan = GraphPlan(pddl)

    def goal_test(kb, goals):
        return all(kb.ask(q) is not False for q in goals)

    goals = expr('RightShoeOn, LeftShoeOn')

    while True:
        if (goal_test(graphplan.graph.levels[-1].kb, goals) and graphplan.graph.non_mutex_goals(goals, -1)):
            solution = graphplan.extract_solution(goals, -1)
            if solution:
                return solution

        graphplan.graph.expand_graph()
        if len(graphplan.graph.levels) >= 2 and graphplan.check_leveloff():
            return None


class TotalOrderPlanner:
    def __init__(self, pddl):
        self.pddl = pddl

    def filter(self, solution):
        """Filter out persistence actions from a solution"""

        new_solution = []
        for section in solution[0]:
            new_section = []
            for operation in section:
                if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                    new_section.append(operation)
            new_solution.append(new_section)
        return new_solution

    def orderlevel(self, level, pddl):
        """Return valid linear order of actions for a given level"""

        for permutation in itertools.permutations(level):
            temp = copy.deepcopy(pddl)
            count = 0
            for action in permutation:
                try:
                    temp.act(action)
                    count += 1
                except:
                    count = 0
                    temp = copy.deepcopy(pddl)
                    break
            if count == len(permutation):
                return list(permutation), temp
        return None

    def execute(self):
        """Finds total-order solution for a planning graph"""

        graphplan_solution = GraphPlan(self.pddl).execute()
        filtered_solution = self.filter(graphplan_solution)
        ordered_solution = []
        pddl = self.pddl
        for level in filtered_solution:
            level_solution, pddl = self.orderlevel(level, pddl)
            for element in level_solution:
                ordered_solution.append(element)

        return ordered_solution


def linearize(solution):
    """Converts a level-ordered solution into a linear solution"""

    linear_solution = []
    for section in solution[0]:
        for operation in section:
            if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                linear_solution.append(operation)

    return linear_solution


def spare_tire_graphplan():
    """Solves the spare tire problem using GraphPlan"""
    return GraphPlan(spare_tire()).execute()

def three_block_tower_graphplan():
    """Solves the Sussman Anomaly problem using GraphPlan"""
    return GraphPlan(three_block_tower()).execute()

def air_cargo_graphplan():
    """Solves the air cargo problem using GraphPlan"""
    return GraphPlan(air_cargo()).execute()

def have_cake_and_eat_cake_too_graphplan():
    """Solves the cake problem using GraphPlan"""
    return [GraphPlan(have_cake_and_eat_cake_too()).execute()[1]]

def shopping_graphplan():
    """Solves the shopping problem using GraphPlan"""
    return GraphPlan(shopping_problem()).execute()

def socks_and_shoes_graphplan():
    """Solves the socks and shoes problem using GraphpPlan"""
    return GraphPlan(socks_and_shoes()).execute()


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


class Problem(PDDL):
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

    def refinements(hla, state, library):  # TODO - refinements may be (multiple) HLA themselves ...
        """
        state is a Problem, containing the current state kb
        library is a dictionary containing details for every possible refinement. eg:
        {
        'HLA': ['Go(Home,SFO)', 'Go(Home,SFO)', 'Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)', 'Taxi(Home, SFO)'],
        'steps': [['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'], ['Taxi(Home, SFO)'], [], [], []],
        # empty refinements ie primitive action
        'precond': [['At(Home), Have(Car)'], ['At(Home)'], ['At(Home)', 'Have(Car)'], ['At(SFOLongTermParking)'], ['At(Home)']],
        'effect': [['At(SFO)'], ['At(SFO)'], ['At(SFOLongTermParking)'], ['At(SFO)'], ['At(SFO)'], ['~At(Home)'], ['~At(Home)'], ['~At(Home)'], ['~At(SFOLongTermParking)'], ['~At(Home)']]
        }
        """
        e = Expr(hla.name, hla.args)
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
        for i in indices:
            # TODO multiple refinements
            precond = []
            for p in library['precond'][i]:
                if p[0] == '~':
                    precond.append(expr('Not' + p[1:]))
                else:
                    precond.append(expr(p))
            effect = []
            for e in library['effect'][i]:
                if e[0] == '~':
                    effect.append(expr('Not' + e[1:]))
                else:
                    effect.append(expr(e))
            action = HLA(library['steps'][i][0], precond, effect)
            if action.check_precond(state.init, action.args):
                yield action

    def hierarchical_search(problem, hierarchy):
        """
        [Figure 11.5] 'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world problem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details)
        """
        act = Node(problem.actions[0])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            print(plan.state.name)
            hla = plan.state  # first_or_null(plan)
            prefix = None
            if plan.parent:
                prefix = plan.parent.state.action  # prefix, suffix = subseq(plan.state, hla)
            outcome = Problem.result(problem, prefix)
            if hla is None:
                if outcome.goal_test():
                    return plan.path()
            else:
                print("else")
                for sequence in Problem.refinements(hla, outcome, hierarchy):
                    print("...")
                    frontier.append(Node(plan.state, plan.parent, sequence))

    def result(problem, action):
        """The outcome of applying an action to the current problem"""
        if action is not None:
            problem.act(action)
            return problem
        else:
            return problem


def job_shop_problem():
    """
    [figure 11.1] JOB-SHOP-PROBLEM

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


class PlanningKB:
    """ A PlanningKB contains a set of Expr objects that are immutable and hashable.
     With its goal clauses and its accompanying h function, the KB
     can be used by the A* algorithm in its search Nodes. (search.py) """
    def __init__(self, goals, initial_clauses=None):
        if initial_clauses is None:
            initial_clauses = []

        initial_clauses = [expr(c) if not isinstance(c, Expr) else c for c in initial_clauses]
        self.clause_set = frozenset(initial_clauses)

        goals = [expr(g) if not isinstance(g, Expr) else g for g in goals]
        self.goal_clauses = frozenset(goals)

    def __eq__(self, other):
        """search.Node has a __eq__ method for each state, so this method must be implemented too."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.clause_set == other.clause_set

    def __lt__(self, other):
        """Goals must be part of each PlanningKB because search.Node has a __lt__ method that compares state to state
        (used for ordering the priority queue). As a result, states must be compared by how close they are to the goal
        using a heuristic."""
        if not isinstance(other, self.__class__):
            return NotImplementedError

        # ordering is whether there are fewer unresolved goals in the current KB than the other KB.
        return len(self.goal_clauses - self.clause_set) < len(self.goal_clauses - other.clause_set)

    def __hash__(self):
        """search.Node has a __hash__ method for each state, so this method must be implemented too.
        Remember that __hash__ requires immutability."""
        return hash(self.clause_set)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, list(self.goal_clauses), list(self.clause_set))

    def goal_test(self):
        """ Goal is satisfied when KB at least contains all goal clauses. """
        return self.clause_set >= self.goal_clauses

    def h(self):
        """ Basic heuristic to return number of remaining goal clauses to be satisfied. Override this with a more
        accurate heuristic, if available."""
        return len(self.goal_clauses - self.clause_set)

    def fetch_rules_for_goal(self, goal):
        return self.clause_set


class PlanningSearchProblem:
    """
    Used to define a planning problem with a non-mutable KB that can be used in a search.
    The states in the knowledge base consist of first order logic statements.
    The conjunction of these logical statements completely define a state.
    """
    def __init__(self, initial_kb, actions):
        self.initial = initial_kb
        self.possible_actions = actions

    @classmethod
    def from_PDDL_object(cls, pddl_obj):
        initial = PlanningKB(pddl_obj.goals, pddl_obj.init)
        planning_actions = []
        for act in pddl_obj.actions:
            planning_actions.append(STRIPSAction.from_action(act))
        return cls(initial, planning_actions)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.initial, self.possible_actions)

    def actions(self, state):
        for action in self.possible_actions:
            for valid, subst in action.check_precond(state):
                if valid:
                    new_action = action.copy()
                    new_action.subst = subst
                    yield new_action

    def goal_test(self, state):
        return state.goal_test()

    def result(self, state, action):
        return action.act(action.subst, state)

    def h(self, node):
        return node.state.h()

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


def is_negative_clause(e):
    return e.op == '~' and len(e.args) == 1


class STRIPSAction:
    """
    Defines an action schema using preconditions and effects
    Use this to describe actions in PDDL
    action is an Expr where variables are given as arguments(args)
    Precondition and effect are both lists with positive and negated literals
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("~Eaten(food)")]
    effect = [expr("Eaten(food)"), expr("~Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, expression, preconds, effects):
        if isinstance(expression, str):
            expression = expr(expression)

        preconds = [expr(p) if not isinstance(p, Expr) else p for p in preconds]
        effects = [expr(e) if not isinstance(e, Expr) else e for e in effects]

        self.name = expression.op
        self.args = expression.args
        self.subst = None
        precond_neg, precond_pos = partition(preconds, is_negative_clause)
        self.precond_pos = set(precond_pos)
        self.precond_neg = set(e.args[0] for e in precond_neg)  # change the negative Exprs to positive for evaluation
        effect_rem, effect_add = partition(effects, is_negative_clause)
        self.effect_add = set(effect_add)
        self.effect_rem = set(e.args[0] for e in effect_rem)  # change the negative Exprs to positive for evaluation

    @classmethod
    def from_action(cls, action):
        op = action.name
        args = action.args
        preconds = []
        for p in action.precond:
            precond_op = p.op.replace('Not', '~')
            precond_args = [repr(a) for a in p.args]
            preconds.append(expr(build_expr_string(precond_op, precond_args)))
        effects = []
        for e in action.effect:
            effect_op = e.op.replace('Not', '~')
            effect_args = [repr(a) for a in e.args]
            effects.append(expr(build_expr_string(effect_op, effect_args)))
        return cls(Expr(op, *args), preconds, effects)

    def __repr__(self):
        preconds = list(self.precond_pos.union(set((expr('~' + repr(p)) for p in self.precond_neg))))
        effects = list(self.effect_add.union(set((expr('~' + repr(e)) for e in self.effect_rem))))
        return '{}({}, {}, {})'.format(self.__class__.__name__, Expr(self.name, *self.args),
                                       preconds, effects)

    def copy(self):
        """ Returns a copy of this object. """
        act = self.__new__(self.__class__)
        act.name = self.name
        act.args = self.args[:]
        act.subst = self.subst
        act.precond_pos = self.precond_pos.copy()
        act.precond_neg = self.precond_neg.copy()
        act.effect_add = self.effect_add.copy()
        act.effect_rem = self.effect_rem.copy()
        return act

    def substitute(self, subst, e):
        """Replaces variables in expression with the same substitution used for the precondition. """
        new_args = [subst.get(x, x) for x in e.args]
        return Expr(e.op, *new_args)

    def check_neg_precond(self, kb, precond, subst):
        if precond:
            found_subst = False
            for s in fol_bc_and(kb, list(precond), subst):
                neg_precond = frozenset(self.substitute(s, x) for x in precond)
                clause_set = kb.fetch_rules_for_goal(None)
                # negative preconditions succeed if none of them are found in the KB.
                found_subst = True
                yield clause_set.isdisjoint(neg_precond), s
            if not found_subst:
                yield True, subst
        else:
            yield True, subst

    def check_pos_precond(self, kb, precond, subst):
        if precond:
            found_subst = False
            for s in fol_bc_and(kb, list(precond), subst):
                pos_precond = frozenset(self.substitute(s, x) for x in precond)
                # are all preconds found in the KB?
                clause_set = kb.fetch_rules_for_goal(None)
                found_subst = True
                yield clause_set.issuperset(pos_precond), s
            if not found_subst:
                yield True, subst
        else:
            yield True, subst

    def check_precond(self, kb):
        """Checks if preconditions are satisfied in the current state"""
        for valid, subst in self.check_pos_precond(kb, self.precond_pos, {}):
            if valid:
                yield from self.check_neg_precond(kb, self.precond_neg, subst)

    def act(self, subst, kb):
        """ Executes the action on a new copy of the PlanningKB """
        new_kb = PlanningKB(kb.goal_clauses, kb.clause_set)
        clause_set = set(new_kb.clause_set)
        neg_literals = set(self.substitute(subst, clause) for clause in self.effect_rem)
        pos_literals = set(self.substitute(subst, clause) for clause in self.effect_add)
        new_kb.clause_set = frozenset(clause_set - neg_literals | pos_literals)
        return new_kb


def print_solution(node):
    if not node or not node.solution():
        print('No solution found.\n')
        return

    for action in node.solution():
        print(action.name, end='(')
        for a in action.args[:-1]:
            print('{},'.format(action.subst.get(a, a)), end=' ')
        if action.args:
            print('{})'.format(action.subst.get(action.args[-1], action.args[-1])))
        else:
            print(')')
    print()


def construct_solution_from_pddl(pddl_domain, pddl_problem) -> None:
    initial_kb = PlanningKB(pddl_problem.goals, pddl_problem.initial_state)
    planning_actions = [STRIPSAction(name, preconds, effects) for name, preconds, effects in pddl_domain.actions]
    p = PlanningSearchProblem(initial_kb, planning_actions)

    print('\n{} solution:'.format(pddl_problem.problem_name))
    print_solution(astar_search(p))


def gather_test_pairs() -> list:
    pddl_dir = os.getcwd() + os.sep + 'pddl_files'
    domain_files = [pddl_dir + os.sep + x for x in os.listdir(pddl_dir) if x.endswith('domain.pddl')]
    problem_files = [pddl_dir + os.sep + x for x in os.listdir(pddl_dir) if x.endswith('problem.pddl')]
    domain_objects = []
    problem_objects = []
    for f in domain_files:
        domain_parser = DomainParser()
        domain_parser.read(f)
        domain_objects.append(domain_parser)

    for f in problem_files:
        problem_parser = ProblemParser()
        problem_parser.read(f)
        problem_objects.append(problem_parser)

    object_pairs = []
    for p in problem_objects:
        for d in domain_objects:
            if p.domain_name == d.domain_name:
                object_pairs.append((d, p))
    if object_pairs:
        return object_pairs
    else:
        raise IOError('No matching PDDL domain and problem files found.')


def test_planning_solutions():
    """ Call this function to run test cases inside PDDL_files directory."""
    for domain, problem in gather_test_pairs():
        construct_solution_from_pddl(domain, problem)
