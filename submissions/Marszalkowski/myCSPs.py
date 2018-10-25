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
    'TH': rgby
}

variables = domains.keys()

neighbors = {
    'TH': ['BW', 'BB', 'RB', 'H'],
    'GF': ['BW', 'BB'],
    'BW': ['GF', 'BB', 'RB', 'H', 'TH'],
    'BB': ['GF', 'BW', 'H', 'BF', 'TH'],
    'RB': ['BW', 'H', 'T', 'MD', 'TH'],
    'MD': ['RB', 'T', 'GHC'],
    'H': ['RB', 'BW', 'BB', 'BF', 'GHC', 'T', 'TH'],
    'BF': ['BB', 'H', 'GHC', 'M'],
    'T': ['MD', 'RB', 'H', 'GHC'],
    'GHC': ['T', 'H', 'BF', 'M', 'MD'],
    'M': ['GHC', 'BF'],
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True


shireCSP = csp.CSP(variables, domains, neighbors, constraints)
shireCSP.label = 'Shire Map'

myCSPs = [
    {
        'csp': shireCSP,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': shireCSP,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': shireCSP,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': shireCSP,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': shireCSP,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp': shireCSP,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]