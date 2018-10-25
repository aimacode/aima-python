import csp

rgby = ['R', 'G', 'B', 'Y']

d2 = { 'N' : rgby, 'NP' : rgby, 'B' : rgby, 'C' : rgby, 'SP' : rgby, 'G' : rgby, 'S' : rgby}

v2 = d2.keys()

# This map cannot be colored using less than four colors.
n2 = {'N' : ['NP', 'B', 'G'],
      'NP' : ['B', 'N'],
      'B' : ['NP', 'N', 'SP', 'C', 'G'],
      'C' : ['B', 'G', 'SP'],
      'SP' : ['B', 'C', 'G'],
      'G' : ['N', 'B', 'C', 'SP', 'S'],
      'S' : ['G']}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Mexico Culinary Map'

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
