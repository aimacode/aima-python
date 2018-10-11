import csp

rgb = ['R', 'G', 'B']
colors = ['R', 'G', 'B', 'Y']

d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

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

possible_colors = {
    '805': colors,
    '661': colors,
    '760': colors,
    '909': colors,
    '951': colors,
    '949': colors,
    '714': colors,
    '562': colors,
    '626': colors,
    '323': colors,
    '213': colors,
    '310': colors,
    '818': colors
}

regions = possible_colors.keys()

area_code_map = {
    '805': ['661', '818', '310'],
    '661': ['760', '626', '818', '805'],
    '760': ['951', '909', '626', '661'],
    '909': ['760', '951', '714', '626'],
    '951': ['760', '949', '714', '909'],
    '949': ['951', '760', '714'],
    '714': ['951', '949', '562', '626', '909'],
    '562': ['626', '714', '310', '323'],
    '626': ['760', '909', '714', '562', '323', '818', '661'],
    '323': ['213', '626', '562', '310', '818'],
    '213': ['323'],
    '310': ['323', '562', '805', '818'],
    '818': ['661', '626', '323', '310', '805']
}

socal = csp.CSP(regions, possible_colors, area_code_map, constraints)
socal.label = 'SoCal Area Codes'


myCSPs = [
    {
        'csp': socal,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': socal,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': socal,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': socal,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': socal,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp': socal,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]

