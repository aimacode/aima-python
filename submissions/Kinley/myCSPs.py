import csp

rgbpw = ['R','G','B', 'P', 'W']

domains = {
    'Argentina': rgbpw,
'Bolivia': rgbpw,
'Brazil': rgbpw,
'Chile': rgbpw,
'Colombia': rgbpw,
'Ecuador': rgbpw,
'French': rgbpw,
'Guyana': rgbpw,
'Paraguay': rgbpw,
'Peru': rgbpw,
'Suriname': rgbpw,
'Uruguay': rgbpw,
'Venezuela': rgbpw,
}

neighbors = {
    'Argentina': ['Uruguay', 'Paraguay', 'Bolivia', 'Chile'],
'Bolivia': ['Brazil', 'Peru', 'Chile', 'Paraguay', 'Argentina'],
'Brazil': ['French', 'Suriname', 'Guyana', 'Venezuela', 'Colombia', 'Peru', 'Bolivia', 'Paraguay', 'Uruguay'],
'Chile': ['Peru', 'Bolivia', 'Argentina',],
'Colombia': ['Venezuela', 'Ecuador',],
'Ecuador': ['Colombia', 'Peru'],
'French': ['Brazil', 'Suriname'],
'Guyana': ['Suriname', 'Venezuela'],
'Paraguy': ['Bolivia', 'Brazil', 'Argentina'],
'Peru': ['Ecuador', 'Colombia', 'Brazil', 'Bolivia', 'Chile'],
'Suriname': ['French', 'Brazil', 'Guyana',],
'Uruguay': ['Brazil', 'Argentina'],
'Venezuela': ['Guyana', 'Brazil', 'Colombia',]
}

variables = neighbors.keys()

domains = {}
for v in variables:
    domains[v] = ['R', 'G', 'B', 'K']

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