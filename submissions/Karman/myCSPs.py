import csp

rgb = ['R', 'G', 'B']

domains = {
    'JF': rgb,
    'BT': rgb,
    'OD': rgb,
    'SH': rgb,
    'SP': rgb,
    'MD': rgb,
    'HD': rgb,
    'NE': rgb,
    'HE': rgb,
    'TR': rgb

}

variables = domains.keys()

neighbors = {
    'JF': ['BT', 'SH', 'OD'],
    'BT': ['NE', 'SP', 'HD'],
    'OD': ['TR', 'HE', 'SH', 'JF'],
    'SH': ['HE', 'SP', 'JF', 'OD'],
    'SP': ['NE', 'JF', 'BT', 'SH'],
    'MD': ['HD'],
    'HD': ['MD', 'BT', 'NE'],
    'NE': ['BT', 'SP', 'HD'],
    'HE': ['TR', 'OD', 'SH'],
    'TR': ['OD', 'HE']
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myKy = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myKy,
     # 'select_unassigned_variable':csp.mrv,
     }
]