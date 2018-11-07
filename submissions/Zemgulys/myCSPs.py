import csp

rgby = ['R', 'G', 'B', 'Y']

d2 = {'U': rgby,
      'P': rgby,
      'V': rgby,
      'A': rgby,
      'K': rgby,
      'S': rgby,
      'T': rgby,
      'M': rgby,
      'TG': rgby,
      'KL': rgby}

v2 = d2.keys()

Lithuania = {'U': ['V', 'P'],
             'P': ['U', 'V', 'K', 'S'],
             'V': ['U', 'P', 'K', 'A'],
             'A': ['V', 'K', 'M'],
             'K': ['A', 'M', 'TG', 'S', 'P', 'V'],
             'S': ['P', 'K', 'TG', 'T'],
             'T': ['S', 'TG', 'KL'],
             'M': ['A', 'K', 'TG'],
             'TG': ['M', 'K', 'S', 'T', 'KL'],
             'KL': ['TG', 'T']}


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, Lithuania, constraints)
c2.label = 'Lithuania Map'

myCSPs = [
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
