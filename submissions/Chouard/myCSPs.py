import csp

rgb = ['R', 'G', 'B']

d2 = {'A': rgb, 'B': rgb, 'C': ['R'], 'D': rgb, }

v2 = d2.keys()

n2 = {'A': ['B', 'C', 'D'],
      'B': ['A', 'C', 'D'],
      'C': ['A', 'B'],
      'D': ['A', 'B'], }


def constraints(A, a, B, b):
    if A == B:  # e.g. NSW == NSW
        return True

    if a == b:  # e.g. WA = G and SA = G
        return False

    return True


c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Really Lame'

colors = ['R', 'G', 'B', 'Y']

domains = {
    'Clatsop': colors, 'Columbia': colors, 'Multnomah': colors, 'Hood River': colors,
    'Wasco': colors, 'Sherman': colors, 'Gilliam': colors, 'Morrow': colors, 'Umatilla': colors,
    'Union': colors, 'Wallowa': colors, 'Tillamook': colors, 'Washington': colors, 'Clackamas': colors,
    'Wheeler': colors, 'Grant': colors, 'Baker': colors, 'Yamhill': colors, 'Marion': colors,
    'Jefferson': colors, 'Polk': colors, 'Lincoln': colors, 'Benton': colors, 'Linn': colors,
    'Deschutes': 'R', 'Crook': colors, 'Lane': colors, 'Douglas': colors, 'Coos': colors,
    'Curry': colors, 'Josephine': colors, 'Jackson': colors, 'Klamath': colors, 'Lake': colors,
    'Harney': colors, 'Malheur': colors
}

variables = domains.keys()

neighbors = {
    'Clatsop': ['Columbia', 'Tillamook'], 'Columbia': ['Clatsop', 'Washington', 'Multnomah'],
    'Multnomah': ['Columbia', 'Washington', 'Hood River', 'Clackamas'],
    'Hood River': ['Multnomah', 'Clackamas', 'Wasco'],
    'Wasco': ['Hood River', 'Clackamas', 'Marion', 'Jefferson', 'Sherman', 'Gilliam', 'Wheeler'],
    'Sherman': ['Wasco', 'Gilliam'],
    'Gilliam': ['Wasco', 'Sherman', 'Wheeler', 'Morrow'], 'Morrow': ['Gilliam', 'Wheeler', 'Grant', 'Umatilla'],
    'Umatilla': ['Morrow', 'Grant', 'Union', 'Wallowa'], 'Wallowa': ['Umatilla', 'Union', 'Baker'],
    'Union': ['Umatilla', 'Wallowa', 'Baker', 'Grant'],
    'Tillamook': ['Clatsop', 'Washington', 'Polk', 'Yamhill', 'Lincoln'],
    'Washington': ['Columbia', 'Multnomah', 'Clackamas', 'Yamhill', 'Tillamook'],
    'Clackamas': ['Washington', 'Yamhill', 'Marion', 'Wasco', 'Hood River', 'Multnomah'],
    'Wheeler': ['Wasco', 'Jefferson', 'Crook', 'Grant', 'Morrow', 'Gilliam'],
    'Grant': ['Morrow', 'Wheeler', 'Crook', 'Harney', 'Malheur', 'Baker', 'Union', 'Umatilla'],
    'Baker': ['Wallowa', 'Union', 'Grant', 'Malheur'],
    'Yamhill': ['Tillamook', 'Polk', 'Marion', 'Clackamas', 'Washington'],
    'Marion': ['Yamhill', 'Polk', 'Linn', 'Jefferson', 'Wasco', 'Clackamas'],
    'Jefferson': ['Wasco', 'Marion', 'Linn', 'Deschutes', 'Crook', 'Wheeler'],
    'Polk': ['Yamhill', 'Tillamook', 'Marion', 'Linn', 'Benton', 'Lincoln'],
    'Lincoln': ['Tillamook', 'Polk', 'Lane', 'Benton'],
    'Benton': ['Polk', 'Lincoln', 'Linn', 'Lane'],
    'Linn': ['Benton', 'Polk', 'Lane', 'Deschutes', 'Jefferson', 'Marion'],
    'Deschutes': ['Linn', 'Lane', 'Klamath', 'Lake', 'Harney', 'Crook', 'Jefferson'],
    'Crook': ['Jefferson', 'Deschutes', 'Harney', 'Grant', 'Wheeler'],
    'Lane': ['Lincoln', 'Benton', 'Linn', 'Deschutes', 'Klamath', 'Douglas'],
    'Douglas': ['Lane', 'Coos', 'Curry', 'Josephine', 'Jackson', 'Klamath'],
    'Coos': ['Douglas', 'Curry'], 'Curry': ['Coos', 'Josephine', 'Douglas'],
    'Josephine': ['Douglas', 'Curry', 'Jackson'], 'Jackson': ['Josephine', 'Douglas', 'Klamath'],
    'Klamath': ['Lane', 'Douglas', 'Jackson', 'Lake', 'Deschutes'],
    'Lake': ['Deschutes', 'Klamath', 'Harney'], 'Harney': ['Grant', 'Crook', 'Deschutes', 'Lake', 'Malheur'],
    'Malheur': ['Baker', 'Grant', 'Harney']
}

oregon = csp.CSP(variables, domains, neighbors, constraints)
oregon.label = 'Oregon Counties'

myCSPs = [
    # {
    #     'csp': c2,
    #     # 'select_unassigned_variable': csp.mrv,
    #     # 'order_domain_values': csp.lcv,
    #     # 'inference': csp.mac,
    #     # 'inference': csp.forward_checking,
    # },
    # {
    #     'csp': c2,
    #     'select_unassigned_variable': csp.mrv,
    #     # 'order_domain_values': csp.lcv,
    #     # 'inference': csp.mac,
    #     # 'inference': csp.forward_checking,
    # },
    # {
    #     'csp': c2,
    #     # 'select_unassigned_variable': csp.mrv,
    #     'order_domain_values': csp.lcv,
    #     # 'inference': csp.mac,
    #     # 'inference': csp.forward_checking,
    # },
    # {
    #     'csp': c2,
    #     # 'select_unassigned_variable': csp.mrv,
    #     # 'order_domain_values': csp.lcv,
    #     'inference': csp.mac,
    #     # 'inference': csp.forward_checking,
    # },
    # {
    #     'csp': c2,
    #     # 'select_unassigned_variable': csp.mrv,
    #     # 'order_domain_values': csp.lcv,
    #     # 'inference': csp.mac,
    #     'inference': csp.forward_checking,
    # },
    # {
    #     'csp': c2,
    #     'select_unassigned_variable': csp.mrv,
    #     'order_domain_values': csp.lcv,
    #     'inference': csp.mac,
    #     # 'inference': csp.forward_checking,
    # },
    {
        'csp': oregon,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': oregon,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': oregon,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': oregon,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp': oregon,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp': oregon,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
