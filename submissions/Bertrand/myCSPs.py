import csp

colors = ['R', 'G', 'B']

domains = {
    'Galicia': colors,
    'Asturias': colors,
    'C': colors,
    'PV': colors,
    'R': colors,
    'N': colors,
    'Aragon': colors,
    'Catalonia': colors,
    'Castile_Leon': colors,
    'Extremadura': colors,
    'Castile_la_Mancha': colors,
    'CV': colors,
    'Andalucia': colors,
    'MU': colors,
    'M': colors
}

neighbors = {
    'Galicia': ['Asturias', 'Castile_Leon'],
    'Asturias': ['C', 'Castile_Leon', 'Galicia'],
    'C': ['Asturias', 'Castile_Leon', 'PV'],
    'PV': ['C', 'Castile_Leon', 'N', 'R'],
    'R': ['Aragon', 'Castile_Leon', 'N', 'PV'],
    'N': ['Aragon', 'PV', 'R'],
    'Aragon': ['Castile_Leon', 'Castile_la_Mancha', 'Catalonia', 'CV', 'N', 'R'],
    'Catalonia': ['Aragon', 'CV'],
    'Castile_Leon': ['Aragon', 'Asturias', 'C', 'Castile_la_Mancha', 'Galicia', 'M', 'PV', 'R'],
    'Extremadura': ['Andalucia', 'Castile_Leon', 'Castile_la_Mancha'],
    'Castile_la_Mancha': ['Andalucia', 'Aragon', 'Castile_Leon', 'CV', 'Extremadura', 'M', 'MU'],
    'CV': ['Aragon', 'Castile_la_Mancha', 'Catalonia', 'MU'],
    'Andalucia': ['Castile_la_Mancha', 'Extremadura', 'MU'],
    'MU': ['Andalucia', 'Castile_la_Mancha', 'CV'],
    'M': ['Castile_Leon', 'Castile_la_Mancha']
}

regions = neighbors.keys()


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(regions, domains, neighbors, constraints)
c2.label = 'Regions of Spain'

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
