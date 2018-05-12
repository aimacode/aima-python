from typing import Deque, AnyStr
from collections import deque

CHAR = 0
WHITESPACE = [' ', '\t']
Symbol = str          # A Lisp Symbol is implemented as a Python str
List = list           # A Lisp List is implemented as a Python list


class ParseError(Exception):
    pass


class Tokens:
    def __init__(self, token_deque, info_deque):
        self.token_deque = token_deque
        self.info_deque = info_deque
        self.last_charstr = ''
        self.last_line = 0
        self.last_col = 0

    def __repr__(self):
        return str(self.token_deque)

    def popleft(self):
        try:
            token = self.token_deque.popleft()
            charstr, line, col = self.info_deque.popleft()
            if type(token) is Deque:
                open_paren = 1
                while open_paren != 0:
                    if charstr == '(':
                        open_paren += 1
                    elif charstr == ')':
                        open_paren -= 1
                    charstr, line, col = self.info_deque.popleft()
            self.last_charstr, self.last_line, self.last_col = charstr, line, col
        except IndexError:
            exc_text = "EOF encountered. Last token processed was '{}' on line {}, col {}."
            raise ParseError(exc_text.format(self.last_charstr, self.last_line, self.last_col))
        return token

    def pop(self):
        try:
            token = self.token_deque.pop()
            charstr, line, col = self.info_deque.pop()
            self.last_charstr, self.last_line, self.last_col = charstr, line, col
        except IndexError:
            raise ParseError("EOF encountered. Last token processed was '{}' " +
                             "on line {}, col {}.".format(self.last_charstr, self.last_line, self.last_col))
        return token

    def lookahead(self, idx=0):
        try:
            return self.token_deque[idx]
        except IndexError:
            return None


def read_pddl_file(filename) -> Tokens:
    with open(filename) as f:
        # read in lines from PDDL file and remove newline characters
        lines = deque([line.strip() for line in f.readlines()])
    strip_comments_and_blank_lines(lines)

    # transform into Python-compatible S-expressions (using deques of strings)
    tokens, info_deque = tokenize(lines)
    token_deque = read_from_tokens(tokens)
    return Tokens(token_deque, info_deque)


def strip_comments_and_blank_lines(lines: deque) -> None:
    """ Given a list of strings, strips any comments. """
    for i, line in enumerate(lines):
        idx = line.find(';')
        if idx != -1:
            lines[i] = line[:idx]

    # remove any blank lines
    for i in range(len(lines)-1, -1, -1):
        if lines[i] == '':
            del lines[i]


def tokenize(lines: deque):
    """Tokenize PDDL contained in a string.
    Add line number and column number info for error reporting."""
    if not lines:
        raise ParseError('No lines in file')
    # join all lines into a single line of PDDL
    pddl = ''.join(lines)
    tokens = deque(pddl.replace('(', ' ( ').replace(')', ' ) ').replace(':', ' :').split())
    token_info = deque()

    # scan lines in file and record placement of each token
    line = lines.popleft()
    line_idx = 0
    curr_col_idx = 0
    for idx, t in enumerate(tokens):
        if not line:
            raise ParseError("Couldn't find token {}".format(t))
        while True:
            col_idx = line.find(t, curr_col_idx)
            if col_idx == -1:
                curr_col_idx = 0
                if not lines:
                    raise ParseError("Couldn't find token {}".format(t))
                line = lines.popleft()
                line_idx += 1
                continue
            else:
                # actual line and col numbers are line_idx+1 and col_idx+1
                token_info.append((t, line_idx+1, col_idx+1))
                curr_col_idx = col_idx + 1
                break
    return tokens, token_info


def read_from_tokens(tokens: deque):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise ParseError('unexpected EOF while reading')
    token = tokens.popleft()
    if '(' == token:
        D = deque()
        try:
            while tokens[0] != ')':
                D.append(read_from_tokens(tokens))
            tokens.popleft()  # pop off ')'
            return D
        except IndexError:
            raise ParseError('unexpected EOF while parsing {}'.format(list(D)))
    elif ')' == token:
        raise ParseError('unexpected ")" token')
    else:
        return token


def parse_tokens(parsers, tokens):
    while tokens:
        for parser in parsers:
            if parser.detect(tokens):
                parser.parse(tokens)
                break
        else:
            # remove a token only when none of the parsers are successful
            tokens.popleft()


class Sequence:
    def __init__(self):
        pass

    def parse(self, tokens: deque):
        token = tokens.popleft()
        if type(token) is not Deque:
            raise ParseError('Expected sequence, but found "{}" instead.'.format(token))


class Define:
    def __init__(self):
        pass

    def detect(self, tokens):
        try:
            return tokens.lookahead() == 'define'
        except IndexError:
            return False

    def parse(self, tokens):
        token = tokens.popleft()
        if token != 'define':
            raise ParseError('Expected "define" keyword at line {}, col {}'.format(tokens.last_line, tokens.last_col))
        return token


class DefineProblem(Define):
    def __init__(self):
        super().__init__()
        self.problem_name = None

    def detect(self, tokens):
        if not super().detect(tokens):
            return False
        try:
            return tokens.lookahead(1)[0] == 'problem' and type(tokens.lookahead(1)[1]) is str
        except IndexError:
            return False
        except TypeError:
            return False

    def parse(self, tokens):
        super().parse(tokens)
        problem_seq = tokens.popleft()
        token = problem_seq.popleft()
        if token != 'problem':
            raise ParseError('Expected "problem" keyword at line {}, col {}'.format(tokens.last_line, tokens.last_col))
        self.problem_name = problem_seq.popleft()


class DefineDomain(Define):
    def __init__(self):
        super().__init__()
        self.domain_name = None

    def detect(self, tokens):
        if not super().detect(tokens):
            return False
        try:
            return tokens.lookahead(1)[0] == 'domain' and type(tokens.lookahead(1)[1]) is str
        except IndexError:
            return False
        except TypeError:
            return False

    def parse(self, tokens):
        if self.domain_name:
            raise ParseError("Domain line occurs twice in domain file.")
        super().parse(tokens)
        domain_seq = tokens.popleft()
        token = domain_seq.popleft()
        if token != 'domain':
            raise ParseError('Expected "domain" keyword at line {}, col {}'.format(tokens.last_line, tokens.last_col))
        self.domain_name = domain_seq.popleft()


class Requirements:
    def __init__(self):
        self.requirements = []

    def detect(self, text):
        try:
            token = text.lookahead()
            if token:
                return token[0].startswith(':requirements')
            else:
                return False
        except IndexError:
            return False

    def parse(self, tokens):
        if self.requirements:
            raise ParseError("Requirements line occurs twice in domain file.")
        token_list = tokens.popleft()
        token_list.popleft()
        while token_list:
            self.requirements.append(token_list.popleft())
        if ':strips' not in self.requirements:
            raise ParseError(':strips is not in list of domain requirements on line {}.'.format(tokens.last_line))


class Variables:
    @classmethod
    def parse(cls, tokens):
        """ Extracts a list of variables from the PDDL. """
        variables = []
        while tokens:
            token = tokens.popleft()
            if not token.startswith('?'):
                raise ParseError("Unrecognized variable name ({0}) " +
                                 "that doesn't begin with a question mark".format(token))
            try:
                pred_var = token[1:]
            except IndexError:
                raise ParseError("Variable name format incorrect")
            # lookahead to see if there's a dash indicating an upcoming type name
            if tokens.lookahead() == '-':
                # get rid of the dash character and the type name
                tokens.popleft()
                tokens.popleft()
            variables.append(pred_var)
        return variables


class Predicate:
    def __init__(self):
        self.expr = None

    def detect(self, tokens):
        return True

    def parse(self, tokens):
        if tokens.lookahead() == 'not':
            # expression is not(e), so next, parse the expression e before prepending the ~ operator to it.
            token = tokens.pop()
            e = cls.parse(token)
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
            self._build_expr_string(expr_name, variables)
            return True

    def _build_expr_string(self, expr_name: str, variables: list) -> str:
        # can't have actions with a dash in the name; it confuses the Expr class
        estr = expr_name.replace('-', '').capitalize() + '('
        vlen = len(variables)
        if vlen:
            for i in range(vlen - 1):
                estr += variables[i] + ', '
            estr += variables[vlen - 1]
        estr += ')'
        self.expr = expr(estr)


class PredicateList:
    def __init__(self):
        pass

    def parse(self, tokens):
        expr_lst = []
        while tokens:
            token = tokens.popleft()
            expr_lst.append(_parse_single_expr_string(token))
        return expr_lst


class Formula:
    def __init__(self):
        pass

    def parse(self, tokens):
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
        parsers = [DefineDomain(), Requirements()]
        parse_tokens(parsers, pddl)


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
        parsers = [DefineProblem()]
        parse_tokens(parsers, pddl)

