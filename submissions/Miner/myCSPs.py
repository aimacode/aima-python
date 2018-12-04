import csp

rgb = ['R', 'G', 'B']

d2 = {'A': rgb, 'B': rgb, 'C': ['R'], 'D': rgb}

v2 = d2.keys()

n2 = {'A': ['B', 'C', 'D'],
      'B': ['A', 'C', 'D'],
      'C': ['A', 'B'],
      'D': ['A', 'B']}


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True


c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Really Lame'

colors = ['R', 'G', 'B', 'Y']

domains = {
    'Riften': colors,
    'Eastmarch': colors,
    'Falkreath': colors,
    'Whiterun': colors,
    'Winterhold': colors,
    'ThePale': colors,
    'Markarth': colors,
    'Solitude': colors,
    'Morthal': colors
}

variables = domains.keys()

neighbors = {
    'Riften': ['Eastmarch', 'Whiterun', 'Falkreath'],
    'Eastmarch': ['Riften', 'Winterhold', 'Whiterun', 'ThePale'],
    'Falkreath': ['Riften', 'Whiterun', 'ThePale', 'Markarth'],
    'Whiterun': ['Riften', 'Falkreath', 'Markarth', 'Eastmarch', 'ThePale', 'Morthal'],
    'Winterhold': ['ThePale', 'Eastmarch'],
    'ThePale': ['Winterhold', 'Morthal', 'Whiterun', 'Eastmarch'],
    'Markarth': ['Falkreath', 'Whiterun', 'Morthal', 'Solitude'],
    'Solitude': ['Markarth', 'Morthal'],
    'Morthal': ['ThePale', 'Whiterun', 'Markarth']
}


skyrim = csp.CSP(variables, domains, neighbors, constraints)
skyrim.label = 'Skyrim Districts'

myCSPs = [
    {
        'csp': skyrim,
    },
    {
        'csp': skyrim,
        'select_unassigned_variable': csp.mrv
    },
    {
        'csp': skyrim,
        'order_domain_values': csp.lcv
    },
    {
        'csp': skyrim,
        'inference': csp.mac
    },
    {
        'csp': skyrim,
        'inference': csp.forward_checking
    },
    {
        'csp': skyrim,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    }
]
