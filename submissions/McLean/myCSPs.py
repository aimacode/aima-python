import csp

rgb = ['R', 'G', 'B']

domains = {
    "Yukon": rgb,
    "Northwest Territories": rgb,
    "Nunavut": rgb,
    "British Columbia": rgb,
    "Alberta": rgb,
    "Saskatchewan": rgb,
    "Manitoba": rgb,
    "Ontario": rgb,
    "Quebec": rgb,
    "Newfoundland": rgb,
    "Prince Edward Island": rgb,
    "New Brunswick": rgb,
    "Nova Scotia": rgb
}

variables = domains.keys()

neighbors = {
    "Yukon": ["Northwest Territories", "British Columbia"],
    "British Columbia": ["Yukon", "Alberta"],
    "Northwest Territories": ["Yukon", "Nunavut", "British Columbia", "Alberta", "Saskatchewan"],
    "Nunavut": ["Northwest Territories", "Quebec", "Manitoba"],
    "Alberta": ["British Columbia", "Northwest Territories", "Saskatchewan"],
    "Saskatchewan": ["Alberta", "Manitoba", "Northwest Territories"],
    "Manitoba": ["Nunavut", "Saskatchewan",  "Ontario"],
    "Ontario": ["Manitoba", "Quebec"],
    "Quebec": ["Ontario", "New Brunswick", "Newfoundland", "Prince Edward Island"],
    "Newfoundland": ["Quebec", "Nova Scotia"],
    "Prince Edward Island": ["New Brunswick", "Nova Scotia"],
    "Nova Scotia": ["New Brunswick", "Prince Edward Island" ],
    "New Brunswick": ["Quebec", "Prince Edward Island", "Nova Scotia"]
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

