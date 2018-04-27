import os
from typing import Deque
from collections import deque

CHAR = 0
WHITESPACE = [' ', '\t']
Symbol = str          # A Lisp Symbol is implemented as a Python str
List = list           # A Lisp List is implemented as a Python list


class ParseError(Exception):
    pass


def read_pddl_file(filename) -> deque:
    with open(filename) as f:
        # read in lines from PDDL file and remove newline characters
        lines = [line.strip() for line in f.readlines()]
    strip_comments(lines)

    # transform into Python-compatible S-expressions (using deques of strings)
    return readlines(filename, lines)


def strip_comments(lines) -> None:
    """ Given a list of strings, strips any comments. """
    for i, line in enumerate(lines):
        idx = line.find(';')
        if idx != -1:
            lines[i] = line[:idx]


def readlines(filename: str, pddl: list) -> deque:
    """Read PDDL contained in a string."""
    return parse(filename, tokenize(pddl))


def parse(filename: str, tokens: deque):
    # read the tokens one at time (left to right) and separate them into list of keywords and parameters.
    if len(tokens) == 0:
        raise ParseError('unexpected EOF while reading {}.'.format(os.path.basename(filename)))
    char, line_no, col_no = tokens.popleft()
    if '(' == char:
        D = deque()
        s = ''
        while True:
            try:
                if tokens[0][CHAR] == '(':
                    D.append(parse(filename, tokens))
                elif tokens[0][CHAR] == ')':
                    if s:
                        D.append(s)
                    tokens.popleft()
                    break
                elif tokens[0][CHAR] in WHITESPACE:
                    if s:
                        D.append(s)
                    tokens.popleft()
                    s = ''
                else:
                    char, line_no, col_no = tokens.popleft()
                    s += char
            except IndexError:
                raise ParseError('unexpected EOF while reading {}.'.format(os.path.basename(filename)))
        return D
    elif ')' == char:
        raise ParseError("unexpected ')' token in {}, line {}, col {}".format(os.path.basename(filename),
                                                                              line_no + 1, col_no + 1))
    else:
        return char


def tokenize(pddl: list) -> deque:
    """Convert a string into a deque of tokens."""
    d = deque()
    for lineno, line in enumerate(pddl):
        for column, char in enumerate(line):
            d.append((char, lineno, column))
    return d


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


def parse_variables(tokens, has_types) -> list:
    """ Extracts a list of variables from the PDDL. """
    variables = []
    while tokens:
        token = tokens.popleft()
        if not token.startswith('?'):
            raise ParseError("Unrecognized variable name ({0}) " +
                             "that doesn't begin with a question mark".format(token))
        pred_var = token[1:]
        if has_types:
            # lookahead to see if there's a dash indicating an upcoming type name
            if tokens[0] == '-':
                # get rid of the dash character and the type name
                tokens.popleft()
                tokens.popleft()
        variables.append(pred_var)
    return variables


def _parse_single_expr_string(tokens: deque) -> str:
    if tokens[0] == 'not':
        # expression is not(e), so next, parse the expression e before prepending the ~ operator to it.
        token = tokens.pop()
        e = _parse_single_expr_string(token)
        if '~' in e:
            raise ParseError('Multiple not operators in expression.')
        return '~' + e
    else:  # expression is a standard Op(param1, param2, etc ...) format
        expr_name = tokens.popleft().capitalize()
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
        expr_lst.append(_parse_single_expr_string(token))
    return expr_lst


def parse_formula(tokens: deque) -> list:
    expr_lst = []
    token = tokens.popleft()
    if token.lower() == 'and':  # preconds and effects only use 'and' keyword
        exprs = _parse_expr_list(tokens)
        expr_lst.extend(exprs)
    else:  # parse single expression
        expr_lst.append(_parse_single_expr_string(deque([token]) + tokens))
    return expr_lst


def build_expr_string(expr_name: str, variables: list) -> str:
    # can't have actions with a dash in the name; it confuses the Expr class
    estr = expr_name.replace('-', '').capitalize() + '('
    vlen = len(variables)
    if vlen:
        for i in range(vlen - 1):
            estr += variables[i] + ', '
        estr += variables[vlen - 1]
    estr += ')'
    return estr


class DomainParser:
    def __init__(self):
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
        domain_seq = tokens.popleft()
        token = domain_seq.popleft()
        if token != 'domain':
            raise ParseError('domain keyword not found after define statement')
            return False
        self.domain_name = domain_seq.popleft()
        return True

    def _parse_requirements(self, tokens: deque) -> bool:
        self._requirements = list(tokens)
        if ':strips' not in self._requirements:
            raise ParseError(':strips is not in list of domain requirements. Cannot parse this domain file.')
        return True

    def _parse_constants(self, tokens: deque) -> bool:
        self.constants = parse_variables(tokens, self._types)
        return True

    # noinspection PyUnusedLocal
    def _parse_types(self, tokens: deque) -> bool:
        self._types = True
        return True

    def _parse_predicates(self, tokens: deque) -> bool:
        while tokens:
            predicate = tokens.popleft()
            pred_name = predicate.popleft()
            new_predicate = [pred_name] + parse_variables(predicate, self._types)
            self.predicates.append(new_predicate)
        return True

    def _parse_action(self, tokens) -> bool:
        self._action_name = tokens.popleft()
        match = {':parameters': self._parse_parameters,
                 ':precondition': self._parse_preconditions,
                 ':effect': self._parse_effects
                 }
        parse_tokens(match, tokens)
        params = [p[0] for p in self._parameters]
        action = (build_expr_string(self._action_name, params), self._preconditions, self._effects)
        self.actions.append(action)
        # reset the temporary storage for this action before processing the next one.
        self._action_name = ''
        self._parameters = []
        self._preconditions = []
        self._effects = []
        return True

    def _parse_parameters(self, tokens: deque) -> bool:
        if tokens:
            param_list = tokens.popleft()
            self._parameters = parse_variables(param_list, self._types)
        return True

    def _parse_preconditions(self, tokens: deque) -> bool:
        if tokens:
            precond_seq = tokens.popleft()
            self._preconditions = parse_formula(precond_seq)
        return True

    def _parse_effects(self, tokens: deque) -> bool:
        if tokens:
            effects_seq = tokens.popleft()
            self._effects = parse_formula(effects_seq)
        return True

    def read(self, filename) -> None:
        pddl = read_pddl_file(filename)

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
        self.problem_name = ''
        self.domain_name = ''
        self.initial_state = []
        self.goals = []

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

    def _parse_init(self, tokens: deque) -> bool:
        self.initial_state = _parse_expr_list(tokens)
        return True

    def _parse_goal(self, tokens: deque) -> bool:
        goal_list = tokens.popleft()
        self.goals = parse_formula(goal_list)
        return True

    def read(self, filename) -> None:
        pddl = read_pddl_file(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':domain': self._parse_domain,
                 ':init': self._parse_init,
                 ':goal': self._parse_goal
                 }

        parse_tokens(match, pddl)
