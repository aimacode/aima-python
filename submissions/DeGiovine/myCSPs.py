import csp

rgby = ['R', 'G', 'B', 'Y']

d2 = { 'A' : rgby, 'B' : rgby, 'C' : ['R'], 'D' : rgby,}

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
c2.label = 'Really Lame'

LowerHudson = {
    'Orange': rgby,
    'Rockland':rgby,
    'Westchester': rgby,
    'Sullivan': rgby,
    'Putnam': rgby,
    'Ulster':rgby,
    'Dutchess': rgby,
    'Delaware': rgby,
    'Greene': rgby,
    'Columbia': rgby,
}

Counties = LowerHudson.keys()

Neighbors ={
    'Orange': ['Sullivan', 'Ulster', 'Dutchess','Putnam', 'Westchester', 'Rockland'],
    'Rockland':['Orange', 'Putnam', 'Westchester'],
    'Westchester': ['Orange','Putnam', 'Rockland'],
    'Sullivan': ['Delaware', 'Ulster', 'Orange'],
    'Putnam': ['Orange','Dutchess','Westchester', 'Rockland'],
    'Ulster': ['Orange', 'Sullivan', 'Delaware','Greene', 'Columbia', 'Dutchess'],
    'Dutchess': ['Columbia', 'Ulster', 'Orange','Putnam'],
    'Delaware': ['Sullivan', 'Ulster', 'Greene'],
    'Greene': ['Delaware', 'Ulster', 'Columbia'],
    'Columbia': ['Greene', 'Ulster', 'Dutchess'],
}

HudsonValley = csp.CSP(Counties, LowerHudson, Neighbors, constraints)
HudsonValley.label = 'Lower Hudson Valley Counties'

myCSPs = [
    {
        'csp' : HudsonValley,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : HudsonValley,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : HudsonValley,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : HudsonValley,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : HudsonValley,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : HudsonValley,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        #'inference': csp.forward_checking,
    },
]
