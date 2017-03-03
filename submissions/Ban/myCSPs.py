import csp

rgb = ['R', 'G', 'B']

domains = {
    'Du': rgb,
    'Dr': rgb,
    'Cv': rgb,
    'Ca': rgb,
    'Mu': rgb,
    'At': rgb,
    'Ki': rgb,
    'Ga': rgb,
}

variables = domains.keys()

neighbors = {
    'Du': ['Dr', 'Mu', 'Ca'],
    'Dr': ['Du', 'Cv'],
    'Cv': ['Dr', 'Mu'],
    'Ca': ['Du', 'Ki', 'At'],
    'Mu': ['Du', 'Cv', 'At'],
    'At': ['Mu', 'Ki', 'Ga'],
    'Ki': ['Ca', 'At'],
    'Ga': ['At']
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myIre = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myIre,
     # 'select_unassigned_variable':csp.mrv,
     }
]