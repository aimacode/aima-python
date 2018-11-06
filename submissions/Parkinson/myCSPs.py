import csp

rgb = ['R', 'G', 'B', 'P']

d2 = { 'Aroostook' : rgb,
       'Washington' : rgb,
       'Hancock' : rgb,
       'Penobscot' : rgb,
       'Piscataquis' : rgb,
       'Somerset' : rgb,
       'Waldo' : rgb,
       'Knox' : rgb,
       'Lincoln' : rgb,
       'Kennebec' : rgb,
       'Franklin' : rgb,
       'Oxford' : rgb,
       'Androscoggin' : rgb,
       'Sagadahoc' : rgb,
       'Cumberland' : rgb,
       'York' : rgb,}

v2 = d2.keys()

n2 = {'Aroostook' : ['Washington', 'Penobscot', 'Piscataquis', 'Somerset'],
      'Washington' : ['Aroostook', 'Penobscot', 'Hancock'],
      'Hancock' : ['Washington', 'Penobscot', 'Waldo', 'Knox'],
      'Penobscot' : ['Hancock', 'Aroostook', 'Piscataquis', 'Washington', 'Waldo', 'Somerset'],
      'Piscataquis' : ['Aroostook', 'Somerset', 'Penobscot'],
      'Somerset' : ['Piscataquis', 'Aroostook', 'Franklin', 'Kennebec', 'Waldo', 'Penobscot'],
      'Waldo' : ['Hancock', 'Penobscot', 'Somerset', 'Knox', 'Lincoln', 'Kennebec'],
      'Knox' : ['Lincoln', 'Waldo', 'Hancock'],
      'Lincoln' : ['Knox', 'Waldo', 'Kennebec', 'Sagadahoc'],
      'Kennebec' : ['Lincoln', 'Waldo', 'Somerset', 'Sagadahoc', 'Androscoggin', 'Franklin'],
      'Franklin' : ['Somerset', 'Kennebec', 'Androscoggin', 'Oxford'],
      'Oxford' : ['Franklin', 'Androscoggin', 'Cumberland', 'York'],
      'Androscoggin' : ['Sagadahoc', 'Kennebec', 'Franklin', 'Oxford', 'Cumberland'],
      'Sagadahoc' : ['Androscoggin', 'Cumberland', 'Kennebec', 'Lincoln'],
      'Cumberland' : ['York', 'Oxford', 'Androscoggin', 'Sagadahoc',],
      'York' : ['Cumberland', 'Oxford']}


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Maine map'

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
