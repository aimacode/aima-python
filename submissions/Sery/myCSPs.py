import csp

rgb = ['R', 'G', 'B']

domains = {
    'krasnoyarsk': rgb,
    'tomsk': rgb,
    'omsk': rgb,
    'novobirsk': rgb,
    'kemerovo': rgb,
    'altai': rgb,
    'khakassia': rgb,
    'tuva': rgb,
    'irkutsk': rgb,
    'buryatia': rgb,
    'chita': rgb,
}

variables = domains.keys()

# Russian Siberian provinces based on http://www.mapsofworld.com/images/russia-political-enlarged-map.jpg
neighbors = {
    'krasnoyarsk': ['tomsk', 'kemerovo', 'khakassia', 'tuva', 'irkutsk'],
    'tomsk': ['krasnoyarsk', 'omsk', 'novobirsk', 'kemerovo'],
    'omsk': ['tomsk', 'novobirsk'],
    'novobirsk': ['omsk', 'tomsk', 'kemerovo', 'altai'],
    'kemerovo': ['krasnoyarsk', 'tomsk', 'novobirsk', 'altai', 'khakassia'],
    'altai': ['novobirsk', 'kemerovo', 'khakassia', 'tuva'],
    'khakassia': ['altai', 'kemerovo', 'krasnoyarsk', 'tuva'],
    'tuva': ['altai', 'khakassia', 'krasnoyarsk', 'irkutsk', 'buryatia'],
    'irkutsk': ['krasnoyarsk', 'tuva', 'buryatia', 'chita'],
    'buryatia': ['tuva', 'irkutsk', 'chita'],
    'chita': ['irkutsk', 'buryatia'],
}


def constraints(A, a, B, b):
    if A == B:  # e.g. NSW == NSW
        return True

    if a == b:  # e.g. WA = G and SA = G
        return False

    return True


myAus = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myAus,
     # 'select_unassigned_variable':csp.mrv,
     }
]
