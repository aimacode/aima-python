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
    return s.replace('(', ' ( ').replace(')', ' ) ').split()


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
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return token
