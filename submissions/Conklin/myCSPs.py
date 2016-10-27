import csp

rgb = ['R', 'G', 'B']

# domains = {
#     'AM': rgb,
#     'ES': rgb,
#     'LK': rgb,
#     'RB': rgb,
#     'FL': rgb,
#     'G': rgb,
#     'S': rgb,
#     'M': rgb,
#     'BL': rgb,
#     'C': rgb,
#     'H': rgb
# }

domains = {
    'Dragonblight': rgb,
    'Borean Tundra': rgb,
    'Scholozar Basin': rgb,
    'Wintergrasp': rgb,
    'Crystalsong': rgb,
    'Icecrown': rgb,
    'Storm Peaks': rgb,
    'Zul Drak': rgb,
    'Grizzly Hills': rgb,
    'Howling Fjord': rgb
}

variables = domains.keys()

# neighbors = {
#     'AM': ['LK', 'ES'],
#     'ES': ['BL', 'M'],
#     'LK': ['RB', 'FL', 'AM'],
#     'RB': ['LK', 'FL', 'H'],
#     'FL': ['G', 'LK', 'RB'],
#     'G': ['FL', 'S'],
#     'S': ['G', 'M'],
#     'M': ['ES', 'BL', 'S'],
#     'BL': ['ES', 'C', 'M'],
#     'C': ['BL', 'H'],
#     'H': ['C', 'RB']
# }

neighbors = {
    'Borean Tundra': {'Scholozar Basin', 'Wintergrasp', 'Dragonblight'},
    'Dragonblight': {'Wintergrasp', 'Icecrown', 'Crystalsong', 'Zul Drak', 'Grizzly Hills'},
    'Wintergrasp': {'Borean Tundra', 'Scholozar Basin', 'Icecrown', 'Dragonblight'},
    'Scholozar Basin': {'Borean Tundra', 'Icecrown', 'Wintergrasp'},
    'Crystalsong': {'Icecrown', 'Dragonblight', 'Storm Peaks', 'Zul Drak'},
    'Icecrown': {'Scholozar Basin', 'Wintergrasp', 'Dragonblight', 'Crystalsong', 'Storm Peaks'},
    'Storm Peaks': {'Icecrown', 'Crystalsong', 'Zul Drak'},
    'Zul Drak': {'Storm Peaks', 'Crystalsong', 'Dragonblight', 'Grizzly Hills'},
    'Grizzly Hills': {'Zul Drak', 'Dragonblight', 'Howling Fjord'},
    'Howling Fjord': {'Grizzly Hills'}
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myNorthrend = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myNorthrend,
     # 'select_unassigned_variable':csp.mrv,
     }
]