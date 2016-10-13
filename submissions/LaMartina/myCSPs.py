import csp

rgb = ['R', 'G', 'B','K']

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

domainsNWAfrica = {
    'WS': rgb,
    'Mor': rgb,
    'Alg': rgb,
    'Tun': rgb,
    'Maur': rgb,
    'Sen': rgb,
    'TheGam': rgb,
    'Gui-Bis': rgb,
    'Gui': rgb,
    'SieLeo': rgb,
    'Lib': rgb,
    'Cot': rgb,
    'Ghana': rgb,
    'Togo': rgb,
    'Benin': rgb,
    'BurFaso': rgb,
    'Mali': rgb
}
variablesAfrica = domainsNWAfrica.keys()

neighborsAfrica = {
    'WS': ['Mor','Maur'],
    'Mor': ['Alg','WS'],
    'Alg': ['Tun','Mali','Mor','Maur','WS'],
    'Tun': ['Alg'],
    'Maur': ['WS','Alg','Mali','Sen'],
    'Sen': ['Mali','Maur','TheGam','Gui-Bis','Gui'],
    'TheGam': ['Sen',],
    'Gui-Bis': ['Sen','Gui'],
    'Gui': ['Gui-Bis','Sen','SieLeo','Lib','Mali','Cot'],
    'SieLeo': ['Gui','Lib'],
    'Lib': ['Gui','SieLeo','Cot'],
    'Cot': ['Lib','Gui','Mali','BurFaso','Ghana'],
    'Ghana': ['Togo','BurFaso','Cot'],
    'Togo': ['BurFaso','Benin','Ghana'],
    'Benin': ['Mali','Togo'],
    'BurFaso': ['Benin','Togo','Ghana','Cot','Mali'],
    'Mali': ['Alg','Maur','Sen','Gui','Cot','BurFaso'],
}

myAfr = csp.CSP(variablesAfrica, domainsNWAfrica, neighborsAfrica, constraints)

myCSPs = [
    {'csp': myAfr,
        #'csp': myAus,
     # 'select_unassigned_variable':csp.mrv,

     }
]