import csp

rgb = ['R', 'G', 'B']

# d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

nigeria2d = { "Lagos" : rgb, "Ogun" : rgb, "Oyo" : rgb, "Osun" : rgb, "Ekiti" : rgb, "Ondo" : rgb, "Kwara" : rgb, "Edo" : rgb,}

# v2 = d2.keys()

nigeria2v = nigeria2d.keys()

# n2 = {'A' : ['B', 'C', 'D'],
#       'B' : ['A', 'C', 'D'],
#       'C' : ['A', 'B'],
#       'D' : ['A', 'B'],}

nigeria2 = {"Lagos" :["Ogun"],
            "Ogun" : ["Lagos", "Oyo", "Osun", "Ondo"],
            "Oyo" :  ["Ogun", "Osun", "Kwara"],
            "Osun" : ["Ogun", "Oyo", "Ekiti", "Kwara", "Ondo"],
            "Ekiti": ["Osun", "Kwara", "Ondo"],
            "Ondo":  ["Ogun", "Osun", "Ekiti", "Edo"],
            "Kwara": ["Oyo", "Osun", "Ekiti"],
            "Edo":   ["Ondo"],}
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
