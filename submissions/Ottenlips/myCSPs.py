import csp

rgb = ['R', 'G', 'B','Y','H']

domains = {
    # 'AM': rgb,
    'Argentina': rgb,
    'Bolivia': rgb,
    'Brazil': rgb,
    'Chile': rgb,
    'Colombia': rgb,
    'Ecuador': rgb,
    'Guyana': rgb,
    'Paraguay': rgb,
    'Peru': rgb,
    'Suriname':rgb,
    'Uruguay':rgb,
    'Venezuela':rgb,
    'French Guiana':rgb

}

variables = domains.keys()

neighbors = {
    'Argentina': ['Bolivia','Brazil','Chile','Uruguay','Paraguay'],
    'Bolivia':['Argentina','Paraguay','Peru','Chile'],
    'Brazil':['Argentina','Uruguay','Paraguay','Bolivia','Peru','Colombia','Venezuela','Suriname','Guyana','French Guiana'],
    'Chile': ['Argentina','Bolivia','Peru'],
    'Colombia': ['Ecuador','Venezuela','Brazil','Peru'],
    'Ecuador': ['Colombia','Peru'],
    'Guyana': ['Venezuela','Suriname','Brazil'],
    'Paraguay': ['Argentina','Bolivia','Brazil'],
    'Peru': ['Ecuador','Colombia','Brazil','Bolivia','Chile'],
    'Suriname': ['Guyana','French Guiana','Brazil'],
    'Uruguay': ['Argentina','Brazil'],
    'Venezuela': ['Columbia','Brazil','Guyana'],
    'French Guiana': ['Brazil','Suriname']

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