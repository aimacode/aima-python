import csp

rgb = ['R', 'G', 'B', 'P']

# d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

nigeria2d = { 'L' :rgb, 'O' : rgb, 'Y' : rgb, 'S' : rgb, 'I' : rgb, 'N' : rgb, 'K' : rgb, 'E' : rgb, 'D' : rgb, 'G' : rgb, 'B' : rgb, 'Ri' : rgb, 'M' : rgb, 'An' : rgb, 'En' : rgb,'Eb' : rgb,'Ab' : rgb, 'Ak' : rgb, 'C' : rgb, 'Be' : rgb,}

# v2 = d2.keys()

nigeria2v = nigeria2d.keys()

# n2 = {'A' : ['B', 'C', 'D'],
#       'B' : ['A', 'C', 'D'],
#       'C' : ['A', 'B'],
#       'D' : ['A', 'B'],}

nigeria2 = {'L':['O'],
            'O': ['L', 'Y', 'S', 'N'],
            'Y': ['O', 'S', 'K'],
            'S': ['O', 'Y', 'I', 'K', 'N'],
            'I': ['S', 'K', 'N', 'G'],
            'N': ['O', 'S', 'I', 'E', 'G'],
            'K': ['Y', 'S', 'I', 'G'],
            'E': ['N', 'G', 'D', 'An'],
            'D': ['E', 'An', 'B', 'Ri'],
            'G': ['E', 'D', 'An', 'En', 'Be', 'I', 'N', 'K'],
            'B': ['D', 'Ri'],
            'Ri': ['B', 'M', 'Ab', 'Ak', 'An', 'D'],
            'M': ['Ri', 'An', 'Ab', 'En'],
            'An': ['G', 'E', 'D', 'Ri', 'M', 'En'],
            'En': ['G', 'An', 'M', 'Ab', 'Eb', 'Be'],
            'Eb': ['En', 'Be', 'Ab', 'C'],
            'Ab': ['C', 'Eb', 'En', 'M', 'Ri', 'Ak'],
            'Ak': ['Ab', 'C', 'Ri'],
            'C': ['Be', 'Eb', 'Ab', 'Ak'],
            'Be': ['G', 'En', 'Eb', 'C'],}

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
