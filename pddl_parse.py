import os
from typing import Deque
from collections import deque


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
    return parse(s, os.path.basename(filename))


def strip_comments(lines) -> None:
    """ Given a list of strings, strips any comments. """
    for i, line in enumerate(lines):
        idx = line.find(';')
        if idx != -1:
            lines[i] = line[:idx]


def parse(pddl, filename):
    """Read PDDL contained in a string."""
    return read_from_tokens(tokenize(pddl), filename)


def tokenize(s: str) -> deque:
    """Convert a string into a list of tokens."""
    return deque(s.replace('(', ' ( ').replace(')', ' ) ').replace(':', ' :').split())


def read_from_tokens(tokens: deque, filename: str):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise ParseError('unexpected EOF while reading {}'.format(filename))
    token = tokens.popleft()
    if '(' == token:
        D = deque()
        try:
            while tokens[0] != ')':
                D.append(read_from_tokens(tokens, filename))
            tokens.popleft()  # pop off ')'
            return D
        except IndexError:
            raise ParseError('unexpected EOF while reading {}'.format(filename))
    elif ')' == token:
        raise ParseError('unexpected ) in {}'.format(filename))
    else:
        return token


def parse_tokens(match_dict, tokens):
    def match_tokens(tokens):
        if not isinstance(tokens, Deque):
            return False
        item = tokens.popleft()
        if isinstance(item, deque):
            match_tokens(item)
        else:
            item = item.lower()
            for text in match_dict:
                if item.startswith(text):
                    if match_dict[text](tokens):
                        break
        return True

    while tokens:
        if not match_tokens(tokens):
            break


def _build_expr_string(expr_name: str, variables: list) -> str:
    # can't have actions with a dash in the name; it confuses the Expr class
    estr = expr_name.replace('-', '').capitalize() + '('
    vlen = len(variables)
    if vlen:
        for i in range(vlen - 1):
            estr += variables[i] + ', '
        estr += variables[vlen - 1]
    estr += ')'
    return estr


def _parse_variables(tokens, has_types) -> list:
    """ Extracts a list of variables from the PDDL. """
    variables = []
    while tokens:
        token = tokens.popleft()
        if token.startswith('?'):
            pred_var = token[1:]
        else:
            pred_var = token

        if has_types:
            # lookahead to see if there's a dash indicating an upcoming type name
            if tokens[0] == '-':
                # get rid of the dash character and the type name
                tokens.popleft()
                tokens.popleft()
        variables.append(pred_var)
    return variables


def _parse_single_expr_string(tokens: deque) -> str:
    if not isinstance(tokens, Deque):
        raise ParseError('Expected expression')
    if tokens[0] == 'not':
        # expression is not(e), so next, parse the expression e before prepending the ~ operator to it.
        token = tokens.pop()
        e = _parse_single_expr_string(token)
        if '~' in e:
            raise ParseError('Multiple not operators in expression.')
        return '~' + e
    else:  # expression is a standard Op(param1, param2, etc ...) format
        expr_name = tokens.popleft().lower()
        variables = []
        while tokens:
            param = tokens.popleft()
            if param.startswith('?'):
                variables.append(param[1:].lower())
            else:
                variables.append(param.capitalize())
        return _build_expr_string(expr_name, variables)


def _parse_expr_list(tokens) -> list:
    if not isinstance(tokens, Deque):
        raise ParseError('Expected expression list')
    expr_lst = []
    while tokens:
        token = tokens.popleft()
        expr_lst.append(_parse_single_expr_string(token))
    return expr_lst


def _parse_formula(tokens: deque) -> list:
    if not isinstance(tokens, Deque) and not tokens:
        raise ParseError('Expected formula')
    expr_lst = []
    token = tokens.popleft()
    if token.lower() == 'and':  # preconds and effects only use 'and' keyword
        exprs = _parse_expr_list(tokens)
        expr_lst.extend(exprs)
    else:  # parse single expression
        expr_lst.append(_parse_single_expr_string(deque([token]) + tokens))
    return expr_lst


class DomainParser:
    def __init__(self):
        self.filename = ''
        self.domain_name = ''
        self._action_name = ''
        self._requirements = []
        self.predicates = []
        self.actions = []
        self.constants = []
        self._types = []
        self._parameters = []
        self._preconditions = []
        self._effects = []

    def _parse_define(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque):
            raise ParseError('Domain list not found after define statement')
        domain_seq = tokens.popleft()
        if not domain_seq:
            raise ParseError('Domain list empty')
        token = domain_seq.popleft()
        if token != 'domain':
            raise ParseError('Domain keyword not found after define statement')
        if not domain_seq:
            raise ParseError('Domain name not found in domain list')
        self.domain_name = domain_seq.popleft()
        return True

    def _parse_requirements(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque):
            raise ParseError('Valid list not found after :requirements keyword')
        self._requirements = list(tokens)
        if ':strips' not in self._requirements:
            raise ParseError(':strips is not in list of domain requirements. ' +
                             'Cannot parse this domain file {}'.format(self.filename))
        return True

    def _parse_constants(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque):
            raise ParseError('Valid list not found after :constants keyword')
        self.constants = _parse_variables(tokens, self._types)
        return True

    # noinspection PyUnusedLocal
    def _parse_types(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque):
            raise ParseError('Expected list of types')
        self._types = True
        return True

    def _parse_predicates(self, tokens: deque) -> bool:
        while tokens:
            if not isinstance(tokens, Deque):
                raise ParseError('Valid list not found after :predicates keyword')
            predicate = tokens.popleft()
            if not isinstance(predicate, Deque):
                raise ParseError('Invalid predicate: {}'.format(predicate))
            pred_name = predicate.popleft()
            if not isinstance(pred_name, str):
                raise ParseError('Invalid predicate name: {}'.format(pred_name))
            if not isinstance(predicate, Deque):
                raise ParseError('Invalid predicate variable list: {}'.format(predicate))
            try:
                new_predicate = [pred_name] + _parse_variables(predicate, self._types)
            except IndexError:
                raise ParseError('Error parsing variables for predicate {}'.format(pred_name))
            self.predicates.append(new_predicate)
        return True

    def _parse_action(self, tokens) -> bool:
        if not isinstance(tokens, Deque):
            raise ParseError('Invalid action: {}'.format(tokens))
        self._action_name = tokens.popleft()
        if not isinstance(self._action_name, str):
            raise ParseError('Invalid action name: {}'.format(self._action_name))

        match = {':parameters': self._parse_parameters,
                 ':precondition': self._parse_preconditions,
                 ':effect': self._parse_effects
                 }
        parse_tokens(match, tokens)
        params = [p for p in self._parameters]
        action = (_build_expr_string(self._action_name, params), self._preconditions, self._effects)
        self.actions.append(action)
        # reset the temporary storage for this action before processing the next one.
        self._action_name = ''
        self._parameters = []
        self._preconditions = []
        self._effects = []
        return True

    def _parse_parameters(self, tokens: deque) -> bool:
        if isinstance(tokens, Deque) and tokens:
            param_list = tokens.popleft()
            if not isinstance(param_list, Deque):
                raise ParseError('Expected parameter list for action {}'.format(self._action_name))
            try:
                self._parameters = _parse_variables(param_list, self._types)
            except IndexError:
                raise ParseError('Error parsing parameter list for action {}'.format(self._action_name))
        return True

    def _parse_preconditions(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque) or not tokens:
            raise ParseError('Invalid precondition list for action {}: {}'.format(self._action_name, tokens))
        precond_seq = tokens.popleft()
        self._preconditions = _parse_formula(precond_seq)
        return True

    def _parse_effects(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque) or not tokens:
            raise ParseError('Invalid effects list for action {}: {}'.format(self._action_name, tokens))
        effects_seq = tokens.popleft()
        self._effects = _parse_formula(effects_seq)
        return True

    def read(self, filename) -> None:
        pddl = read_pddl_file(filename)
        self.filename = os.path.basename(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':requirements': self._parse_requirements,
                 ':constants': self._parse_constants,
                 ':types': self._parse_types,
                 ':predicates': self._parse_predicates,
                 ':action': self._parse_action
                 }

        parse_tokens(match, pddl)


class ProblemParser:
    def __init__(self):
        self.filename = ''
        self.problem_name = ''
        self.domain_name = ''
        self.initial_state = []
        self.goals = []

    def _parse_define(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque) or not tokens:
            raise ParseError('Expected problem list after define statement')
        problem_seq = tokens.popleft()
        if not isinstance(problem_seq, Deque) or not problem_seq:
            raise ParseError('Problem list empty')
        token = problem_seq.popleft()
        if token != 'problem':
            raise ParseError('Problem keyword not found after define statement')
        self.problem_name = problem_seq.popleft()
        return True

    def _parse_domain(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque) or not tokens:
            raise ParseError('Expected domain name')
        self.domain_name = tokens.popleft()

    def _parse_init(self, tokens: deque) -> bool:
        self.initial_state = _parse_expr_list(tokens)
        return True

    def _parse_goal(self, tokens: deque) -> bool:
        if not isinstance(tokens, Deque) or not tokens:
            raise ParseError('Valid list not found after :goal keyword')
        goal_list = tokens.popleft()
        self.goals = _parse_formula(goal_list)
        return True

    def read(self, filename) -> None:
        pddl = read_pddl_file(filename)
        self.filename = os.path.basename(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':domain': self._parse_domain,
                 ':init': self._parse_init,
                 ':goal': self._parse_goal
                 }

        parse_tokens(match, pddl)

