import csp

rgb = ['R', 'G', 'B', 'P']

# d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

nigeria2d = { 'L' :['P'], 'O' : rgb, 'Y' : rgb, 'S' : rgb, 'I' : rgb, 'N' : rgb, 'K' : rgb, 'E' : rgb,}

# v2 = d2.keys()

nigeria2v = nigeria2d.keys()

# n2 = {'A' : ['B', 'C', 'D'],
#       'B' : ['A', 'C', 'D'],
#       'C' : ['A', 'B'],
#       'D' : ['A', 'B'],}

nigeria2 = {'L' :['O'],
            'O' : ['L', 'Y', 'S', 'N'],
            'Y' :  ['O', 'S', 'K'],
            'S' : ['O', 'Y', 'I', 'K', 'N'],
            'I': ['S', 'K', 'N'],
            'N':  ['O', 'S', 'I', 'E'],
            'K': ['Y', 'S', 'I'],
            'E':   ['N'],}
def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False
    return True

# c2 = csp.CSP(v2, d2, n2, constraints)
# c2.label = 'Really Lame'

nigeria = csp.CSP(nigeria2v, nigeria2d, nigeria2, constraints)
nigeria.label = "Simplified Map of Nigeria"

myCSPs = [
    {
        'csp' : nigeria,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : nigeria,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : nigeria,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : nigeria,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : nigeria,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : nigeria,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
