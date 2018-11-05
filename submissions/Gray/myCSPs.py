import csp

rgb = ['R', 'G', 'B', 'P']

d2 = { 'BC' : rgb, 'Alberta' : rgb, 'Saska' : ['R'],
           'Manitoba' : rgb, 'Ontario' : rgb, 'Quebec' : rgb,
           'NewBrunswick' : rgb, 'NovaScotia' : rgb,
           'Newfoundland' : rgb, 'Yukon' : rgb, 'NW' : rgb,
           'Nunavut' : rgb }

v2 = d2.keys()

n2 = {'BC' : ['Yukon', 'NW', 'Alberta'],
      'Alberta' : ['BC', 'Saska', 'NW'],
      'Saska' : ['Alberta', 'NW', 'Nunavut', 'Manitoba'],
      'Manitoba' : ['Saska', 'Nunavut', 'Ontario', 'NW'],
      'Ontario' : ['Manitoba', 'Quebec'],
      'Quebec' : ['Ontario', 'Newfoundland', 'NewBrunswick'],
      'NewBrunswick' : ['Quebec', 'NovaScotia', 'Newfoundland'],
      'NovaScotia' : ['NewBrunswick', 'Newfoundland'],
      'Newfoundland' : ['Quebec', 'NovaScotia', 'NewBrunswick'],
      'Yukon' : ['BC', 'NW'],
      'NW' : ['Yukon', 'BC', 'Alberta', 'Nunavut', 'Saska', 'Manitoba'],
      'Nunavut' : ['NW', 'Manitoba', 'Saska']}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Canada Map'

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
