import csp

rgb = ['R', 'G', 'B']

domains = {
    'AM': rgb,
    'ES': rgb,
    'LK': rgb,
    'RB': rgb,
    'FL': rgb,
    'G': rgb,
    'S': rgb,
    'M': rgb,
    'BL': rgb,
    'C': rgb,
    'H': rgb
}

variables = domains.keys()

neighbors = {
    'AM': ['LK', 'ES'],
    'ES': ['BL', 'M'],
    'LK': ['RB', 'FL', 'AM'],
    'RB': ['LK', 'FL', 'H'],
    'FL': ['G', 'LK', 'RB'],
    'G': ['FL', 'S'],
    'S': ['G', 'M'],
    'M': ['ES', 'BL', 'S'],
    'BL': ['ES', 'C', 'M'],
    'C': ['BL', 'H'],
    'H': ['C', 'RB']
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myAus = csp.CSP(variables, domains, neighbors, constraints)

domainsAfrica = {
    'WS': rgb,
    'Mor': rgb,
    'Alg': rgb,
    'Tun': rgb,
    'Maur': rgb,
    'Sen': rgb,
    'TheGam': rgb,
    'Gui-Bis': rgb,
    ''

}
myCSPs = [
    {'csp': myAus,
     # 'select_unassigned_variable':csp.mrv,
     }
]