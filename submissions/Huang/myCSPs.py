import csp

rgb = ['R', 'G', 'B']

d2 = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

v2 = d2.keys()

n2 = {       'ZB': ['HK', 'HP', 'JA', 'BS', 'PT'],
             'HK': ['YP', 'PD', 'ZB', 'BS'],
             'JA': ['ZB', 'HP', 'PT', 'CN', 'XH'],
             'FX': ['JS', 'SJ', 'PD', 'MH'],
             'XH': ['HP', 'JA', 'CN', 'PD', 'MH'],
             'HP': ['ZB', 'HK', 'PD', 'JA', 'XH'],
             'CN': ['JA', 'PT', 'JD', 'MH', 'XH'],
             'PT': ['ZB', 'JA', 'CN', 'BS', 'JD'],
             'YP': ['HK', 'BS', 'PD'],
             'SJ': ['MH', 'QP', 'JD', 'FX'],
             'QP': ['JD', 'MH', 'SJ'],
             'JD': ['BS', 'PT', 'CN', 'MH', 'QP'],
             'JS': ['SJ', 'QP', 'FX'],
             'PD': ['FX', 'MH', 'XH', 'HP', 'HK', 'YP'],
             'MH': ['XH', 'CN', 'FX', 'JD', 'QP', 'SJ'],
             'BS': ['YP', 'PT', 'JD', 'ZB', 'HK'],
             }

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
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
