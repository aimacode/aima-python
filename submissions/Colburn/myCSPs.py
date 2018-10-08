import csp

rgb = ['R', 'G', 'B','O']

d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

domains = {
    'AM': 'G',
    'ES': rgb,
    'LK': rgb,
    'RB': rgb,
    'FL': rgb,
    'G': rgb,
    'S': rgb,
    'M': rgb,
    'BL': rgb,
    'C': rgb,
    'H': rgb
}

variables = domains.keys()

neighbors = {
    'AM': ['LK', 'ES'],
    'ES': ['BL', 'M'],
    'LK': ['RB', 'FL', 'AM'],
    'RB': ['LK', 'FL', 'H'],
    'FL': ['G', 'LK', 'RB'],
    'G': ['FL', 'S'],
    'S': ['G', 'M'],
    'M': ['ES', 'BL', 'S'],
    'BL': ['ES', 'C', 'M'],
    'C': ['BL', 'H'],
    'H': ['C', 'RB']
}


v2 = d2.keys()

n2 = {'A' : ['B', 'C', 'D'],
      'B' : ['A', 'C', 'D'],
      'C' : ['A', 'B'],
      'D' : ['A', 'B'],}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Map of UK'

UK=csp.CSP(variables,domains,neighbors,constraints)
UK.label = "Map of the UK"

myCSPs = [
    {
        'csp': UK,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    }
    ,
    {
        'csp' : UK,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : UK,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : UK,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : UK,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : UK,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    }

]
