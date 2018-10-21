import csp

#https://geology.com/county-map/texas.shtml link for map

rgb = ['R', 'G', 'B']

d2 = { 'Dallas' : rgb, 'Rockwall' : rgb, 'Kaufman' : rgb, 'Ellis' : rgb, 'Johnson' : rgb,
       'Tarrant' : rgb, 'Denton' : rgb, 'Collin' : rgb, 'Hunt' : rgb}

d3 = { 'B' : ['B'], 'Y' : ['Y'], 'R' : rgb, 'G' : rgb}

v2 = d2.keys()

v3 = d3.keys()

n2 = {'Dallas' : ['Rockwall', 'Kaufman', 'Ellis', 'Tarrant', 'Denton', 'Collin',],
      'Rockwall' : ['Dallas', 'Collin', 'Hunt', 'Kaufman', ],
      'Kaufman' : ['Ellis', 'Dallas', 'Rockwall', 'Hunt',],
      'Ellis' : ['Kaufman', 'Johnson', 'Tarrant','Dallas', ],
      'Johnson' : ['Tarrant', 'Ellis',],
      'Tarrant' : ['Dallas', 'Denton', 'Johnson', 'Ellis',],
      'Denton' : ['Tarrant', 'Collin', 'Dallas',],
      'Collin' : ['Hunt', 'Dallas', 'Denton', 'Rockwall',],
      'Hunt' : ['Rockwall', 'Kaufman', 'Collin',],}

c3 = {'B' : ['G', 'Y', 'R',],
      'Y' : ['B', 'G', 'R',],
      'R' : ['G', 'Y', 'B',],
      'G' : ['Y', 'B', 'R',],}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = Gw
        return False

    return True

# swap comments bellow to swap between texas 3-color and 4 color map

#c2 = csp.CSP(v2, d2, n2, constraints)
#c2.label = 'Texas map'

c2 = csp.CSP(v3, d3, c3, constraints)
c2.label = '4 Color Map'

myCSPs = [
    {
        'csp' : c2,
        # 'csp2' : c3,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        # 'csp2' : c3,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'csp2' : c3,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'csp2' : c3,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'csp2' : c3,
        'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'csp2' : c3,
        # 'inference': csp.forward_checking,
    },
]
