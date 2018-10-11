import csp

rgb = ['R', 'G', 'B','O']#,'White','Gray','Y','Purple','Brown','seafoam','T','Kale']

d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

domains = {
    'SW': ['G'],
    'SE': rgb,
     'L': rgb,
    'EE': rgb,
     'W': rgb,
    'WM': rgb,
    'EM': rgb,
    'NW': rgb,
    'YH': rgb,
    'NE': rgb,
    'S': rgb,

}

variables = domains.keys()

neighbors = {
    'SW': ['SE','WM','W'],
    'SE': ['SW','L','EE','EM','WM'],
     'L': ['SE','EE'],
    'EE': ['SE','EM','L'],
     'W': ['SW','WM','NW'],
    'WM': ['SW','SE','W','EM','NW'],
    'EM': ['WM','NW','YH','SE','EE'],
    'NW': ['W','WM','S','NE','YH'],
    'YH': ['NW','EM','NE'],
    'NE': ['S','NW','YH'],
    'S':  ['NE','NW'],
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
c2.label = 'Really Lame'

UK=csp.CSP(variables,domains,neighbors,constraints)
UK.label = "Map of the Uk"

myCSPs = [
    {
        'csp': c2,
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
