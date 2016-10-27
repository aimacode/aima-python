import csp

rgb = ['R', 'G', 'B']

domains = {         # GERMANY
    'BW': rgb,      # Baden-Wurttemburg
    'BY': rgb,      # Bavaria
    'BE': rgb,      # Berlin
    'SN': rgb,      # Saxony
    'HH': rgb,      # Hamburg
    'HB': rgb,      # Bremen
    'NW': rgb,      # North Rhine-Westphalia
}

variables = domains.keys()

neighbors = {
    'BW': ['BY'],
    'BY': ['BW', 'SN'],
    'BE': [],
    'SN': ['BY'],
    'HH': ['HB'],
    'HB': ['HH'],
    'NW': [],
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myGer = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myGer,
     # 'select_unassigned_variable':csp.mrv,
     }
]