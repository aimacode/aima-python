import csp

rgb = ['R', 'G', 'B']

domains = {
    # Norway
    'NO': rgb,
    # Sweden
    'SW': rgb,
    # Finland
    'FL': rgb,
    # Denmark
    'DM': rgb,
    # Estonia
    'ES': rgb,
    # Latvia
    'LA': rgb,


}

variables = domains.keys()

neighbors = {
    'NO': ['DM', 'SW',],
    'SW': ['NO', 'DM', 'FL'],
    'FL': ['SW', 'ES'],
    'DM': ['NO', 'SW'],
    'ES': ['FL', 'LA'],
    'LA': ['ES']
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