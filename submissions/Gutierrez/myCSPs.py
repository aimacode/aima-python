import csp

rgb = ['R', 'G', 'B','O']

d2 = { 'Racine' : ['R'], 'Kenosha' : rgb, 'Walworth' : rgb, 'Waukesha' : rgb,'Washington' : rgb, 'Ozaukee': rgb,
       'Rock': rgb, 'Jefferson': rgb, 'Dodge':rgb ,'Milwaukee' : rgb}

v2 = d2.keys()

wisconsin2d = {'Milwaukee': ['Racine', 'Waukesha', 'Ozaukee'],
      'Racine' : ['Milwaukee', 'Waukesha', 'Kenosha','Walworth'],
      'Kenosha' : ['Racine', 'Walworth'],
      'Walworth' : ['Kenosha', 'Racine','Waukesha','Jefferson','Rock'],
      'Waukesha':['Walworth','Jefferson','Dodge','Washington','Milwaukee','Racine'],
      'Washington':['Waukesha','Dodge','Ozaukee'],
      'Ozaukee': ['Washington','Milwaukee'],
      'Rock':['Walworth','Jefferson'],
      'Jefferson':['Rock','Walworth','Waukesha','Dodge',],
      'Dodge':['Washington','Waukesha','Jefferson'],}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, wisconsin2d, constraints)
c2.label = 'Really Lame'

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
