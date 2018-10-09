import csp

rgby = ['R', 'G', 'B', 'Y']

domains = {
    'GF': rgby,
    'BW': rgby,
    'BB': rgby,
    'RB': rgby,
    'MD': rgby,
    'H': rgby,
    'BF': rgby,
    'T': rgby,
    'GHC': rgby,
    'M': rgby,
}

variables = domains.keys()

neighbors = {
    'GF': ['BW', 'BB'],
    'BW': ['GF', 'BB', 'RB', 'H'],
    'BB': ['GF', 'BW', 'H', 'BF'],
    'RB': ['BW', 'H', 'T', 'MD'],
    'MD': ['RB', 'T'],
    'H': ['RB', 'BW', 'BB', 'BF', 'GHC', 'T'],
    'BF': ['BB', 'H', 'GHC', 'M'],
    'T': ['MD', 'RB', 'H', 'GHC'],
    'GHC': ['T', 'H', 'BF', 'M'],
    'M': ['GHC', 'BF']
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

ShireCSP = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': ShireCSP,
     'select_unassigned_variable':csp.mrv,
     }
]