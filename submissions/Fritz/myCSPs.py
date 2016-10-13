import csp

rgbpw = ['R', 'G', 'B', 'P', 'W']

domains = {
    'VE': rgbpw,
    'CO': rgbpw,
    'EC': rgbpw,
    'P': rgbpw,
    'B': rgbpw,
    'GU': rgbpw,
    'S': rgbpw,
    'FG': rgbpw,
    'BO': rgbpw,
    'CH': rgbpw,
    'AR': rgbpw,
    'PG': rgbpw,
    'UG': rgbpw
}

variables = domains.keys()

neighbors = {
    'VE': ['CO', 'B', 'GU'],
    'CO': ['EC', 'P', 'B', 'VE'],
    'EC': ['CO', 'P'],
    'P': ['EC', 'CO', 'B', 'BO'],
    'B': ['VE', 'CO', 'P', 'BO', 'P', 'FG', 'S', 'GU'],
    'GU': ['VE', 'B', 'S'],
    'S': ['GU', 'B', 'FG'],
    'FG': ['S' ,'B'],
    'BO': ['B', 'P', 'CH', 'AR', 'PG'],
    'CH': ['BO', 'AR'],
    'AR': ['CH', 'BO', 'PG', 'UG'],
    'PG': ['B', 'BO', 'AR', 'UG'],
    'UG': ['AR', 'PG']
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