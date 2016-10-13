import csp

rgb = ['R', 'G', 'B']

domains = {
    'WA': rgb,
    'NT': rgb,
    'SA': rgb,
    'Q': rgb,
    'NSW': rgb,
    'V': rgb,
    'T': rgb
}

variables = domains.keys()

neighbors = {
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'Q': ['NT', 'SA', 'NSW'],
    'NSW': ['Q', 'SA', 'V'],
    'V': ['SA', 'NSW'],
    'T': []
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myAus = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myAus,
     # 'select_unassigned_variable':csp.mrv,
     }
]