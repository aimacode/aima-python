import csp

rgb = ['R', 'G', 'B']

domains = {
    'Ar' : rgb,
    'Dk' : rgb,
    'Ab' : rgb,
    'Tx' : rgb,
    'Kk' : rgb,
    'Lz' : rgb,
    'Ha' : rgb,
    'Il' : rgb,
}

variables = domains.keys()

neighbors = {
    'Ar' : ['Dk', 'Kk', 'Tx'],
    'Dk' : ['Ar', 'Ab'],
    'Ab' : ['Dk', 'Kk'],
    'Tx' : ['Ar', 'Lz', 'Ha'],
    'Kk' : ['Ar', 'Lz', 'Ab'],
    'Lz' : ['Kk', 'Tx', 'Ha', 'Il'],
    'Ha' : ['Tx', 'Lz'],
    'Il' : ['Lz'],
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myMap = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myMap,
     # 'select_unassigned_variable':csp.mrv,
     }
]