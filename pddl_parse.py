from typing import Deque
from collections import deque
from planning import PlanningAction
from utils import expr


Symbol = str          # A Lisp Symbol is implemented as a Python str
List = list           # A Lisp List is implemented as a Python list


class ParseError(Exception):
    pass


def read_pddl_file(filename) -> list:
    with open(filename) as f:
        # read in lines from PDDL file and remove newline characters
        lines = [line.strip() for line in f.readlines()]
    strip_comments(lines)
    # join all lines into single string
    s = ''.join(lines)
    # transform into Python-compatible S-expressions (using lists of strings)
    return parse(s)


def strip_comments(lines) -> None:
    """ Given a list of strings, strips any comments. """
    for i, line in enumerate(lines):
        idx = line.find(';')
        if idx != -1:
            lines[i] = line[:idx]


def parse(pddl):
    """Read PDDL contained in a string."""
    return read_from_tokens(tokenize(pddl))


def tokenize(s: str) -> deque:
    """Convert a string into a list of tokens."""
    return deque(s.replace('(', ' ( ').replace(')', ' ) ').replace(':', ' :').split())


def read_from_tokens(tokens: deque):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.popleft()
    if '(' == token:
        D = deque()
        while tokens[0] != ')':
            D.appendleft(read_from_tokens(tokens))
        tokens.popleft()  # pop off ')'
        return D
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return token


def parse_tokens(match_dict, token_list):
    def match_tokens(tokens):
        if not tokens:
            return False
        item = tokens.popleft()
        if isinstance(item, Deque):
            match_tokens(item)
        else:
            item = item.lower()
            for text in match_dict:
                if item.startswith(text):
                    if match_dict[text](tokens):
                        break
        return True

    while True:
        if not match_tokens(token_list):
            break


def parse_variables(tokens, types) -> list:
    """ Extracts a list of variables from the PDDL. """
    variables = []
    while tokens:
        token = tokens.popleft()
        if not token.startswith('?'):
            raise ParseError("Unrecognized variable name ({0}) " +
                             "that doesn't begin with a question mark".format(token))
        pred_var = token[1:]
        if types:
            # lookahead to see if there's a dash indicating an upcoming type name
            if tokens[0] == '-':
                # get rid of the dash character and the type name
                tokens.popleft()
                tokens.popleft()
        variables.append(pred_var)
    return variables


def _parse_single_expr_string(tokens: deque) -> str:
    if tokens[0] == 'not':
        token = tokens.popleft()
        e = _parse_single_expr_string(token)
        if '~' in e:
            raise ParseError('Multiple not operators in expression.')
        return '~' + e
    else:
        expr_name = tokens.popleft().lower()
        variables = []
        while tokens:
            param = tokens.popleft()
            if param.startswith('?'):
                variables.append(param[1:].lower())
            else:
                variables.append(param)
        return build_expr_string(expr_name, variables)


def _parse_expr_list(tokens) -> list:
    expr_lst = []
    while tokens:
        token = tokens.popleft()
        e = _parse_single_expr_string(token)
        expr_lst.append(expr(e))
    return expr_lst


def parse_formula(tokens: deque) -> list:
    expr_lst = []
    token = tokens.popleft()
    if token.lower() == 'and':  # preconds and effects only use 'and' keyword
        exprs = _parse_expr_list(tokens)
        expr_lst.extend(exprs)
    else:  # parse single expression
        e = _parse_single_expr_string([token] + tokens)
        expr_lst.append(expr(e))
    return expr_lst


def build_expr_string(expr_name: str, variables: list) -> str:
    estr = expr_name + '('
    vlen = len(variables)
    if vlen:
        for i in range(vlen - 1):
            estr += variables[i] + ', '
        estr += variables[vlen - 1]
    estr += ')'
    return estr


class PDDLDomainParser:
    def __init__(self):
        self.domain_name = ''
        self.action_name = ''
        self.requirements = []
        self.predicates = []
        self.actions = []
        self.types = []
        self.constants = []
        self.parameters = []
        self.preconditions = []
        self.effects = []

    def _parse_define(self, tokens: deque) -> bool:
        domain_list = tokens.popleft()
        token = domain_list.popleft()
        if token != 'domain':
            raise ParseError('domain keyword not found after define statement')
        self.domain_name = domain_list.popleft()
        return True

    def _parse_requirements(self, tokens: deque) -> bool:
        self.requirements = list(tokens)
        if ':strips' not in self.requirements:
            raise ParseError(':strips is not in list of domain requirements. Cannot parse this domain file.')
        return True

    def _parse_constants(self, tokens: deque) -> bool:
        self.constants = parse_variables(tokens)
        return True

    def _parse_types(self, tokens: deque) -> bool:
        self.types = True
        return True

    def _parse_predicates(self, tokens: deque) -> bool:
        while tokens:
            predicate = tokens.popleft()
            pred_name = predicate.popleft()
            new_predicate = [pred_name] + parse_variables(predicate, self.types)
            self.predicates.append(new_predicate)
        return True

    def _parse_action(self, tokens) -> bool:
        self.action_name = tokens.pop()
        self.parameters = []
        self.preconditions = []
        self.effects = []
        match = {':parameters': self._parse_parameters,
                 ':precondition': self._parse_precondition,
                 ':effect': self._parse_effect
                 }
        parse_tokens(match, tokens)
        params = [p[0] for p in self.parameters]
        action_str = build_expr_string(self.action_name, params)
        action = PlanningAction(expr(action_str), self.preconditions, self.effects)
        self.actions.append(action)
        return True

    def _parse_parameters(self, tokens: deque) -> bool:
        param_list = tokens.popleft()
        self.parameters = parse_variables(param_list, self.types)
        return True

    def _parse_precondition(self, tokens: deque) -> bool:
        precond_list = tokens.popleft()
        self.preconditions = parse_formula(precond_list)
        return True

    def _parse_effect(self, tokens: deque) -> bool:
        effects_list = tokens.popleft()
        self.effects = parse_formula(effects_list)
        return True

    def read(self, filename) -> None:
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

        parse_tokens(match, pddl_list)


class PDDLProblemParser:
    def __init__(self, types):
        self.problem_name = ''
        self.types = types
        self.objects = []
        self.init = []
        self.goal = []

    def _parse_define(self, tokens: deque) -> bool:
        problem_list = tokens.popleft()
        token = problem_list.popleft()
        if token != 'problem':
            raise ParseError('problem keyword not found after define statement')
        self.problem_name = problem_list.popleft()
        return True

    def _parse_domain(self, tokens: deque) -> bool:
        self.domain_name = tokens.popleft()
        return True

    def _parse_init(self, tokens: deque):
        self.initial_kb = _parse_expr_list(tokens)
        return True

    def _parse_goal(self, tokens: deque):
        goal_list = tokens.popleft()
        self.goal = parse_formula(goal_list)
        return True

    def read(self, filename):
        pddl_list = read_pddl_file(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':domain': self._parse_domain,
                 ':init': self._parse_init,
                 ':goal': self._parse_goal
                 }

        parse_tokens(match, pddl_list)
