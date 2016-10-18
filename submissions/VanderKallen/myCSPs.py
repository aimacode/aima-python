import csp

rgbp = ['R', 'G', 'B',]

domains = {
    'Solitude': rgbp,
    'Morthal': rgbp,
    'Dawnstar': rgbp,
    'Winterhold': rgbp,
    'Windhelm': rgbp,
    'Whiterun': rgbp,
    'Markarth': rgbp,
    'Falkreath': rgbp,
    'Riften': rgbp,
    'Bruma': rgbp,
    'Chorrol': rgbp,
    'Cheydinhal': rgbp,
    'Imperial City': rgbp,
    'Kvatch': rgbp,
    'Anvil': rgbp,
    'Skingrad': rgbp,
    'Bravil': rgbp,
    'Leyawiin': rgbp,
}

variables = domains.keys()

neighbors = {
'Solitude': ['Morthal', 'Markarth'],
    'Morthal': ['Solitude', 'Dawnstar', 'Whiterun', 'Markarth'],
    'Dawnstar': ['Morthal', 'Whiterun', 'Windhelm', 'Winterhold'],
    'Winterhold': ['Dawnstar', 'Windhelm'],
    'Windhelm': ['Winterhold', 'Dawnstar', 'Whiterun', 'Riften'],
    'Whiterun': ['Dawnstar', 'Windhelm', 'Markarth', 'Riften', 'Falkreath', 'Morthal'],
    'Markarth': ['Solitude', 'Morthal', 'Whiterun', 'Falkreath'],
    'Falkreath': ['Markarth', 'Whiterun', 'Riften', 'Bruma'],
    'Riften': ['Whitehelm', 'Whiterun', 'Falkreath', 'Bruma', 'Cheydinhal'],
    'Bruma': ['Falkreath', 'Riften', 'Cheydinhal', 'Chorral'],
    'Chorrol': ['Bruma', 'Cheydinhal', 'Imperial City', 'Skingrad', 'Kvatch'],
    'Cheydinhal': ['Bruma', 'Riften', 'Bravil', 'Chorrol'],
    'Imperial City': ['Chorrol', 'Skingrad'],
    'Kvatch': ['Chorrol', 'Skingrad', 'Anvil'],
    'Anvil': ['Kvatch'],
    'Skingrad': ['Imperial City', 'Chorrol', 'Kvatch'],
    'Bravil': ['Cheydinhal', 'Leyawiin'],
    'Leyawiin': ['Bravil'],
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myAus = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myAus,
     # 'select_unassigned_variable':csp.mrv,
     }
]