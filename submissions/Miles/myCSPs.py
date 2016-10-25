import csp

rgb = ['R', 'G', 'B']

domains = {
    'BHM': rgb,
    'MON': rgb,
    'HNT': rgb,
    'MB': rgb,
    'AU': rgb,
    'TSC': rgb,
    'FH': rgb,
    'GND': rgb,
}

variables = domains.keys()

neighbors = {
    'BHM': ['TSC', 'AU', 'HNT'],
    'MON': ['AU', 'MB', 'FH'],
    'HNT': ['BHM', 'GND'],
    'MB': ['MON', 'FH'],
    'AU': ['BHM', 'MONT', 'TSC'],
    'TSC': ['AU', 'BHM'],
    'FH': ['MB', 'MON'],
    'GND': ['HNT'],
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