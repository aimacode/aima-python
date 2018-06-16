from collections.abc import MutableSequence
from planning import STRIPSAction
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


def tokenize(s):
    """Convert a string into a list of tokens."""
    return s.replace('(', ' ( ').replace(')', ' ) ').replace(':', ' :').split()


def read_from_tokens(tokens):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        # reverse each list so we can continue to use .pop() on it, and the elements will be in order.
        L.reverse()
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return token


def parse_tokens(match_dict, token_list):
    def match_tokens(tokens):
        if not tokens:
            return False
        item = tokens.pop()
        if isinstance(item, MutableSequence):
            match_tokens(item)
        else:
            for text in match_dict:
                if item.startswith(text):
                    if match_dict[text](tokens):
                        break
        return True

    while True:
        if not match_tokens(token_list):
            break


def parse_variables(tokens, types) -> list:
    variables = []
    num_tokens = len(tokens)
    idx = 0
    while idx < num_tokens:
        if not tokens[idx].startswith('?'):
            raise ParseError("Unrecognized variable name ({0}) " +
                             "that doesn't begin with a question mark".format(tokens[idx]))
        pred_var = tokens[idx][1:]
        if not types:
            variables.append(pred_var)
            idx += 1
        else:
            # lookahead to see if there's a dash indicating an upcoming type name
            if tokens[idx + 1] == '-':
                pred_type = tokens[idx + 2].lower()
                if pred_type not in types:
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


def build_expr_string(expr_name, variables):
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
        self.constants = parse_variables(tokens, self.types)
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
            pred_name = predicate.pop()
            predicate.reverse()
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
        action = STRIPSAction(action_str, self.preconditions, self.effects)
        self.actions.append(action)
        return True

    def _parse_parameters(self, tokens) -> bool:
        param_list = tokens.pop()
        param_list.reverse()
        self.parameters = parse_variables(param_list, self.types)
        return True

    def _parse_single_expr_string(self, tokens) -> str:
        if tokens[0] == 'not':
            token = tokens.pop()
            token.reverse()
            e = self._parse_single_expr_string(token)
            if '~' in e:
                raise ParseError('Multiple not operators in expression.')
            return '~' + e
        else:
            expr_name = tokens[0]
            variables = []
            idx = 1
            num_tokens = len(tokens)
            while idx < num_tokens:
                param = tokens[idx]
                if param.startswith('?'):
                    variables.append(param[1:].lower())
                else:
                    variables.append(param)
                idx += 1
            return build_expr_string(expr_name, variables)

    def _parse_expr_list(self, tokens) -> list:
        expr_lst = []
        while tokens:
            token = tokens.pop()
            token.reverse()
            e = self._parse_single_expr_string(token)
            expr_lst.append(expr(e))
        return expr_lst

    def _parse_formula(self, tokens) -> list:
        expr_lst = []
        token = tokens.pop()
        if token == 'and':  # preconds and effects only use 'and' keyword
            exprs = self._parse_expr_list(tokens)
            expr_lst.extend(exprs)
        else:  # parse single expression
            e = self._parse_single_expr_string([token] + tokens)
            expr_lst.append(expr(e))
        return expr_lst

    def _parse_precondition(self, tokens) -> bool:
        precond_list = tokens.pop()
        self.preconditions = self._parse_formula(precond_list)
        return True

    def _parse_effect(self, tokens) -> bool:
        effects_list = tokens.pop()
        self.effects = self._parse_formula(effects_list)
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

    def _parse_define(self, tokens) -> bool:
        problem_list = tokens.pop()
        token = problem_list.pop()
        if token != 'problem':
            raise ParseError('problem keyword not found after define statement')
        self.problem_name = problem_list.pop()
        return True

    def _parse_objects(self, tokens) -> bool:
        self.objects = parse_variables(tokens)
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
            pred_name = predicate.pop()
            predicate.reverse()
            new_predicate = [pred_name] + self._parse_variables(predicate)
            self.predicates.append(new_predicate)
        return True

    def _parse_variables(self, tokens) -> list:
        variables = []
        num_tokens = len(tokens)
        idx = 0
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
        action = STRIPSAction(action_str, self.preconditions, self.effects)
        self.actions.append(action)
        return True

    def _parse_parameters(self, tokens) -> bool:
        param_list = tokens.pop()
        param_list.reverse()
        self.parameters = self._parse_variables(param_list)
        return True

    def _parse_single_expr_string(self, tokens):
        if tokens[0] == 'not':
            token = tokens.pop()
            token.reverse()
            e = self._parse_single_expr_string(token)
            if '~' in e:
                raise ParseError('Multiple not operators in expression.')
            return '~' + e
        else:
            expr_name = tokens[0]
            variables = []
            idx = 1
            num_tokens = len(tokens)
            while idx < num_tokens:
                param = tokens[idx]
                if param.startswith('?'):
                    variables.append(param[1:].lower())
                else:
                    variables.append(param)
                idx += 1
            return build_expr_string(expr_name, variables)

    def _parse_expr_list(self, tokens):
        expr_lst = []
        while tokens:
            token = tokens.pop()
            token.reverse()
            e = self._parse_single_expr_string(token)
            expr_lst.append(expr(e))
        return expr_lst

    def _parse_formula(self, tokens):
        expr_lst = []
        token = tokens.pop()
        if token == 'and':  # preconds and effects only use 'and' keyword
            exprs = self._parse_expr_list(tokens)
            expr_lst.extend(exprs)
        else:  # parse single expression
            e = self._parse_single_expr_string([token] + tokens)
            expr_lst.append(expr(e))
        return expr_lst

    def _parse_precondition(self, tokens):
        precond_list = tokens.pop()
        self.preconditions = self._parse_formula(precond_list)
        return True

    def _parse_effect(self, tokens):
        effects_list = tokens.pop()
        self.effects = self._parse_formula(effects_list)
        return True

    def read(self, filename):
        pddl_list = read_pddl_file(filename)

        # Use dictionaries for parsing. If the token matches the key, then call the associated value (method)
        # for parsing.
        match = {'define': self._parse_define,
                 ':domain': self._parse_domain,
                 ':objects': self._parse_objects,
                 ':init': self._parse_init,
                 ':goal': self._parse_goal
                 }

        parse_tokens(match, pddl_list)
