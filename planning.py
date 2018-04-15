"""Planning (Chapters 10-11)
"""
from logic import fol_bc_and
from utils import expr, Expr, partition
from search import astar_search
from parse import read_pddl_file, ParseError
from collections.abc import MutableSequence


class PlanningKB:
    """ A PlanningKB contains a set of Expr objects that are immutable and hashable.
     With its goal clauses and its accompanying h function, the KB
     can be used by the A* algorithm in its search Nodes. (search.py) """
    def __init__(self, goals, initial_clauses=None):
        if initial_clauses is None:
            initial_clauses = []
        self.goal_clauses = frozenset(goals)
        self.clause_set = frozenset(initial_clauses)

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

        # heuristic is whether there are fewer unresolved goals in the current KB than the other KB.
        return len(self.goal_clauses - self.clause_set) < len(self.goal_clauses - other.clause_set)

    def __hash__(self):
        """search.Node has a __hash__ method for each state, so this method must be implemented too."""
        return hash(self.clause_set)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, list(self.goal_clauses), list(self.clause_set))

    def goal_test(self):
        """ Goal is satisfied when KB at least contains all goal clauses. """
        return self.clause_set >= self.goal_clauses

    def h(self):
        """ Returns: number of remaining goal clauses to be satisfied """
        return len(self.goal_clauses - self.clause_set)

    def fetch_rules_for_goal(self, goal):
        return self.clause_set


class PlanningProblem:
    """
    Used to define a planning problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely define a state.
    """
    def __init__(self, initial_kb, actions):
        self.initial = initial_kb
        self.possible_actions = actions

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.initial, self.possible_actions)

    def actions(self, state):
        for action in self.possible_actions:
            for subst in action.check_precond(state):
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


class PlanningAction:
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
        self.expression = expression
        self.name = expression.op
        self.args = expression.args
        self.subst = None
        self.preconds = preconds
        self.effects = effects

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, Expr(self.name, *self.args),
                                       list(self.preconds), list(self.effects))

    def copy(self):
        """ Returns a copy of this object. """
        act = self.__new__(self.__class__)
        act.name = self.name
        act.args = self.args[:]
        act.subst = self.subst
        act.preconds = self.preconds.copy()
        act.effects = self.effects.copy()
        return act

    def substitute(self, subst, e):
        """Replaces variables in expression with the same substitution used for the precondition. """
        new_args = [subst.get(x, x) for x in e.args]
        return Expr(e.op, *new_args)

    def check_neg_precond(self, kb, precond, subst):
        for s in subst:
            for _ in fol_bc_and(kb, list(precond), s):
                # if any negative preconditions are satisfied by the substitution, then exit loop.
                if precond:
                    break
            else:
                neg_precond = frozenset(self.substitute(s, x) for x in precond)
                clause_set = kb.fetch_rules_for_goal(None)
                # negative preconditions succeed if none of them are found in the KB.
                if clause_set.isdisjoint(neg_precond):
                    yield s

    def check_pos_precond(self, kb, precond, subst):
        clause_set = kb.fetch_rules_for_goal(None)
        for s in fol_bc_and(kb, list(precond), subst):
            pos_precond = frozenset(self.substitute(s, x) for x in precond)
            # are all preconds found in the KB?
            if clause_set.issuperset(pos_precond):
                yield s

    def check_precond(self, kb):
        """Checks if preconditions are satisfied in the current state"""
        precond_neg, precond_pos = partition(self.preconds, is_negative_clause)
        precond_neg = set(e.args[0] for e in precond_neg)  # change the negative Exprs to positive
        yield from self.check_neg_precond(kb, precond_neg, self.check_pos_precond(kb, precond_pos, {}))

    def act(self, subst, kb):
        """ Executes the action on a new copy of the PlanningKB """
        new_kb = PlanningKB(kb.goal_clauses, kb.clause_set)
        neg_effects, pos_effects = partition(self.effects, is_negative_clause)
        neg_effects = set(self.substitute(subst, e.args[0]) for e in neg_effects)
        pos_effects = set(self.substitute(subst, e) for e in pos_effects)
        new_kb.clause_set = frozenset(kb.clause_set - neg_effects | pos_effects)
        return new_kb


class PDDLDomainParser:
    def __init__(self):
        self.domain_name = ''
        self.action_name = ''
        self.tokens = []
        self.requirements = []
        self.predicates = []
        self.actions = []
        self.types = []
        self.constants = []
        self.parameters = []
        self.preconditions = []
        self.effects = []

    def _parse_define(self, tokens) -> bool:
        domain_list = tokens.pop()
        token = domain_list.pop()
        if token != 'domain':
            raise ParseError('domain keyword not found after define statement')
        self.domain_name = domain_list.pop()
        return True

    def _parse_requirements(self, tokens) -> bool:
        self.requirements = tokens
        if ':strips' not in self.requirements:
            raise ParseError(':strips is not in list of domain requirements. Cannot parse this domain file.')
        return True

    def _parse_constants(self, tokens) -> bool:
        self.constants = self._parse_variables(tokens)
        for const, ctype in self.constants:
            if ctype not in self.types:
                raise ParseError('Constant type {0} not found in list of valid types'.format(ctype))
        return True

    def _parse_types(self, tokens) -> bool:
        self.types = tokens
        return True

    def _parse_predicates(self, tokens) -> bool:
        while tokens:
            predicate = tokens.pop()
            predicate.reverse()
            new_predicate = [predicate[0]] + self._parse_variables(predicate)
            self.predicates.append(new_predicate)
        return True

    def _parse_variables(self, tokens) -> list:
        variables = []
        num_tokens = len(tokens)
        idx = 1
        while idx < num_tokens:
            if not tokens[idx].startswith('?'):
                raise ParseError("Unrecognized variable name ({0}) " +
                                 "that doesn't begin with a question mark".format(tokens[idx]))
            pred_var = tokens[idx][1:]
            if not self.types:
                variables.append(pred_var)
                idx += 1
            else:
                # lookahead to see if there's a dash indicating an upcoming type name
                if tokens[idx + 1] == '-':
                    pred_type = tokens[idx + 2].lower()
                    if pred_type not in self.types:
                        raise ParseError("Predicate type {0} not in type list.".format(pred_type))
                else:
                    pred_type = None
                arg = [pred_var, pred_type]
                variables.append(arg)
                # if any immediately prior variables didn't have an assigned type, then assign them this one.
                for j in range(len(variables) - 1, 0, -1):
                    if variables[j][1] is not None:
                        break
                    else:
                        variables[j][1] = pred_type
                idx += 3
        return variables

    def _parse_action(self, tokens) -> bool:
        self.action_name = self.tokens[idx].lower()
        idx += 1
        match = {':parameters': self._parse_parameters,
                 ':precondition': self._parse_precondition,
                 ':effect': self._parse_effect
                 }
        idx = self.match_and_parse_tokens(idx, match)
        return True

    def _parse_parameters(self, tokens) -> bool:
        idx += 1
        if self.tokens[idx] != '(':
            raise IOError('Start of parameter list is missing an open parenthesis.')
        self.parameters.clear()
        while idx < self.num_tokens:
            if self.tokens[idx] == ')':
                self.num_parens -= 1
                break
            elif self.tokens[idx] == '(':
                self.num_parens += 1
                try:
                    param_vars, idx = self._parse_variables(idx+1)
                except IOError:
                    raise IOError('Action name {0} has an invalid argument list.'.format(self.action_name))
                self.parameters.extend(param_vars)
        return True

    def _parse_single_expr(self, idx):
        if self.tokens[idx+1] == 'not':
            e = self._parse_single_expr(idx + 2)
            if '~' in e:
                raise IOError('Multiple not operators in expression.')
            return expr('~' + e)
        else:
            if self.tokens[idx] != '(':
                raise IOError('Expression in {0} is missing an open parenthesis.'.format(self.action_name))
            while idx < self.num_tokens:
                if self.tokens[idx] == ')':
                    self.num_parens -= 1
                    idx += 1
                    break
                elif self.tokens[idx] == '(':
                    expr_name = self.tokens[idx + 1]
                    variables = []
                    idx += 2
                    while idx < self.num_tokens:
                        if self.tokens[idx] == ')':
                            self.num_parens -= 1
                            break
                        param = self.tokens[idx]
                        if param.startswith('?'):
                            variables.append(param.lower())
                        else:
                            variables.append(param)
            estr = expr_name + '('
            vlen = len(variables)
            for i in range(vlen - 1):
                estr += variables[i] + ', '
            estr += variables[vlen-1] + ')'
            return estr

    def _parse_expr_list(self, idx):
        expr_lst = []
        while idx < self.num_tokens:
            if self.tokens[idx] == ')':
                self.num_parens -= 1
                break
            elif self.tokens[idx] == '(':
                idx, expr = self._parse_single_expr(idx)
                expr_lst.append(expr)
            idx += 1
        return expr_lst

    def _parse_formula(self, idx, label):
        expr_lst = []
        idx += 1
        if self.tokens[idx] == '(':
            self.num_parens += 1
        else:
            raise IOError('Start of {0} {1} is missing an open parenthesis.'.format(self.action_name, label))
        if self.tokens[idx + 1] == 'and':  # preconds and effects only use 'and' keyword
            exprs = self._parse_expr_list(idx + 2)
            expr_lst.extend(exprs)
        else:  # parse single expression
            expr = self._parse_single_expr(idx + 2)
            expr_lst.append(expr)
        return expr_lst

    def _parse_precondition(self, tokens):
        idx, self.preconditions = self._parse_formula(idx, 'preconditions')
        return True

    def _parse_effect(self, tokens):
        idx, self.effects = self._parse_formula(idx, 'effects')
        return True

    def read(self, filename):
        pddl_list = read_pddl_file(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':requirements': self._parse_requirements,
                 ':constants': self._parse_constants,
                 ':types': self._parse_types,
                 ':predicates': self._parse_predicates,
                 ':action': self._parse_action
                 }

        def parse_tokens(tokens):
            if not tokens:
                return
            item = tokens.pop()
            if isinstance(item, MutableSequence):
                parse_tokens(item)
            else:
                for text in match:
                    if item.startswith(text):
                        if match[text](tokens):
                            break

        while True:
            parse_tokens(pddl_list)


def print_solution(node):
    for action in node.solution():
        print(action.name, end='(')
        for a in action.args[:-1]:
            print('{},'.format(action.subst.get(a, a)), end=' ')
        if action.args:
            print('{})'.format(action.subst.get(action.args[-1], action.args[-1])))
        else:
            print(')')
    print()


def air_cargo():
    goals = [expr('At(C1, JFK)'), expr('At(C2, SFO)')]

    init = PlanningKB(goals,
                      [expr('At(C1, SFO)'),
                       expr('At(C2, JFK)'),
                       expr('At(P1, SFO)'),
                       expr('At(P2, JFK)'),
                       expr('Cargo(C1)'),
                       expr('Cargo(C2)'),
                       expr('Plane(P1)'),
                       expr('Plane(P2)'),
                       expr('Airport(JFK)'),
                       expr('Airport(SFO)')])

    # Actions
    #  Load
    precond = [expr('At(c, a)'), expr('At(p, a)'), expr('Cargo(c)'), expr('Plane(p)'), expr('Airport(a)')]
    effect = [expr('In(c, p)'), expr('~At(c, a)')]
    load = PlanningAction(expr('Load(c, p, a)'), precond, effect)

    #  Unload
    precond = [expr('In(c, p)'), expr('At(p, a)'), expr('Cargo(c)'), expr('Plane(p)'), expr('Airport(a)')]
    effect = [expr('At(c, a)'), expr('~In(c, p)')]
    unload = PlanningAction(expr('Unload(c, p, a)'), precond, effect)

    #  Fly
    #  Used used 'f' instead of 'from' because 'from' is a python keyword and expr uses eval() function
    precond = [expr('At(p, f)'), expr('Plane(p)'), expr('Airport(f)'), expr('Airport(to)')]
    effect = [expr('At(p, to)'), expr('~At(p, f)')]
    fly = PlanningAction(expr('Fly(p, f, to)'), precond, effect)

    p = PlanningProblem(init, [load, unload, fly])
    print_solution(astar_search(p))


def spare_tire():
    goals = [expr('At(Spare, Axle)')]
    init = PlanningKB(goals,
                      [expr('At(Flat, Axle)'),
                       expr('At(Spare, Trunk)')])
    # Actions
    #  Remove(Spare, Trunk)
    precond = [expr('At(Spare, Trunk)')]
    effect = [expr('At(Spare, Ground)'), expr('~At(Spare, Trunk)')]
    remove_spare = PlanningAction(expr('Remove(Spare, Trunk)'), precond, effect)
    #  Remove(Flat, Axle)
    precond = [expr('At(Flat, Axle)')]
    effect = [expr('At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    remove_flat = PlanningAction(expr('Remove(Flat, Axle)'), precond, effect)
    #  PutOn(Spare, Axle)
    precond = [expr('At(Spare, Ground)'), expr('~At(Flat, Axle)')]
    effect = [expr('At(Spare, Axle)'), expr('~At(Spare, Ground)')]
    put_on_spare = PlanningAction(expr('PutOn(Spare, Axle)'), precond, effect)
    #  LeaveOvernight
    precond = []
    effect = [expr('~At(Spare, Ground)'), expr('~At(Spare, Axle)'), expr('~At(Spare, Trunk)'),
              expr('~At(Flat, Ground)'), expr('~At(Flat, Axle)')]
    leave_overnight = PlanningAction(expr('LeaveOvernight'), precond, effect)
    p = PlanningProblem(init, [remove_spare, remove_flat, put_on_spare, leave_overnight])
    print_solution(astar_search(p))


def three_block_tower():
    goals = [expr('On(A, B)'), expr('On(B, C)')]
    init = PlanningKB(goals,
                      [expr('On(A, Table)'),
                       expr('On(B, Table)'),
                       expr('On(C, Table)'),
                       expr('Block(A)'),
                       expr('Block(B)'),
                       expr('Block(C)'),
                       expr('Clear(A)'),
                       expr('Clear(B)'),
                       expr('Clear(C)')])

    # Actions
    #  Move(b, x, y)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Clear(y)'), expr('Block(b)')]
    effect = [expr('On(b, y)'), expr('Clear(x)'), expr('~On(b, x)'), expr('~Clear(y)')]
    move = PlanningAction(expr('Move(b, x, y)'), precond, effect)

    #  MoveToTable(b, x)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    effect = [expr('On(b, Table)'), expr('Clear(x)'), expr('~On(b, x)')]
    move_to_table = PlanningAction(expr('MoveToTable(b, x)'), precond, effect)

    p = PlanningProblem(init, [move, move_to_table])
    print_solution(astar_search(p))


def sussman_anomaly():
    goals = [expr('On(A, B)'), expr('On(B, C)')]
    init = PlanningKB(goals,
                      [expr('On(A, Table)'),
                       expr('On(B, Table)'),
                       expr('On(C, A)'),
                       expr('Block(A)'),
                       expr('Block(B)'),
                       expr('Block(C)'),
                       expr('Clear(B)'),
                       expr('Clear(C)')])

    # Actions
    #  Move(b, x, y)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Clear(y)'), expr('Block(b)')]
    effect = [expr('On(b, y)'), expr('Clear(x)'), expr('~On(b, x)'), expr('~Clear(y)')]
    move = PlanningAction(expr('Move(b, x, y)'), precond, effect)

    #  MoveToTable(b, x)
    precond = [expr('On(b, x)'), expr('Clear(b)'), expr('Block(b)')]
    effect = [expr('On(b, Table)'), expr('Clear(x)'), expr('~On(b, x)')]
    move_to_table = PlanningAction(expr('MoveToTable(b, x)'), precond, effect)

    p = PlanningProblem(init, [move, move_to_table])
    print_solution(astar_search(p))


def put_on_shoes():
    goals = [expr('On(RightShoe, RF)'), expr('On(LeftShoe, LF)')]
    init = PlanningKB(goals, [expr('Clear(LF)'),
                              expr('Clear(RF)'),
                              expr('LeftFoot(LF)'),
                              expr('RightFoot(RF)')])

    # Actions
    #  RightShoe
    precond = [expr('On(RightSock, x)'), expr('RightFoot(x)'), expr('~On(RightShoe, x)')]
    effect = [expr('On(RightShoe, x)')]
    right_shoe = PlanningAction(expr('RightShoeOn'), precond, effect)

    #  RightSock
    precond = [expr('Clear(x)'), expr('RightFoot(x)')]
    effect = [expr('On(RightSock, x)'), expr('~Clear(x)')]
    right_sock = PlanningAction(expr('RightSockOn'), precond, effect)

    #  LeftShoe
    precond = [expr('On(LeftSock, x)'), expr('LeftFoot(x)'), expr('~On(LeftShoe, x)')]
    effect = [expr('On(LeftShoe, x)')]
    left_shoe = PlanningAction(expr('LeftShoeOn'), precond, effect)

    #  LeftSock
    precond = [expr('Clear(x)'), expr('LeftFoot(x)')]
    effect = [expr('On(LeftSock, x)'), expr('~Clear(x)')]
    left_sock = PlanningAction(expr('LeftSockOn'), precond, effect)

    p = PlanningProblem(init, [right_shoe, right_sock, left_shoe, left_sock])
    print_solution(astar_search(p))


def parse_domain_file(filename):
    parser = PDDLDomainParser()
    parser.read(filename)


def tester():
    print('Air cargo solution:')
    air_cargo()
    print('\nSpare tire solution:')
    spare_tire()
    print('\nThree block tower solution:')
    three_block_tower()
    print('\nSussman anomaly solution:')
    sussman_anomaly()
    print('\nPut on shoes solution:')
    put_on_shoes()
    parse_domain_file('blocks-domain.pddl')


if __name__ == '__main__':
    tester()
