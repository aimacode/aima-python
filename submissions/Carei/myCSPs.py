import csp

provinces = {'AH': ['JS', 'JX', 'HB', 'HA', 'SD'],
             'BJ': ['HE', 'TJ'],
             'CQ': ['SC', 'SN', 'HB', 'HN', 'GZ'],
             'FJ': ['GD', 'JX', 'ZJ'],
             'GD': ['GX', 'HN', 'JX', 'FJ'],
             'GS': ['XJ', 'QH', 'NM', 'NX', 'SX'],
             'GX': ['YN', 'GZ', 'HN', 'GD'],
             'GZ': ['YN', 'SC', 'CQ', 'HN', 'GX'],
             'HA': ['HB', 'SN', 'SX', 'HE', 'SD', 'AH'],
             'HB': ['HN', 'CQ', 'SN', 'HA', 'AH', 'JX'],
             'HE': ['BJ', 'TJ', 'LN', 'NM', 'SX', 'HA', 'SD'],
             'HI': [],
             'HK': ['GD'],
             'HL': ['NM', 'JL'],
             'HN': ['GX', 'GZ', 'CQ', 'HB', 'JX'],
             'JL': ['HL', 'NM', 'LN'],
             'JS': ['SD', 'AH', 'ZJ'],
             'JX': ['GD', 'HN', 'HB', 'AH', 'ZJ', 'FJ'],
             'LN': ['JL', 'NM', 'HE'],
             'NM': ['GS', 'NX', 'SN', 'LN', 'JL', 'HL'],
             'NX': ['GS', 'NM', 'SN'],
             'QH': ['XJ', 'XZ', 'GS', 'SC'],
             'SC': ['XZ', 'QH', 'GS', 'SN', 'CQ', 'GZ', 'YN'],
             'SD': ['HE', 'HA', 'AH', 'JS'],
             'SH': ['JS', 'ZJ'],
             'SN': ['GS', 'NM', 'SX', 'HN', 'HB', 'CQ', 'SC', 'NX'],
             'SX': ['SN', 'NM', 'HE', 'HA', 'CQ', 'SC'],
             'TJ': ['BJ', 'HE'],
             'TW': [],
             'XJ': ['GS', 'QH', 'XZ'],
             'XZ': ['XJ', 'QH', 'SC', 'YN'],
             'YN': ['XZ', 'SC', 'GZ', 'GX'],
             'ZJ': ['SH', 'JS', 'JX', 'FJ'],

}

# provinces = {'AH': ['JS', 'ZJ', 'JX', 'HB', 'HA', 'SD'],
#              'BJ': ['HE', 'TJ'],
#              'CQ': ['SC', 'SN', 'HB', 'HN', 'GZ'],
#              'FJ': ['GD', 'JX', 'ZJ'],
#              'GD': ['GX', 'HN', 'JX', 'FJ'],
#              'GS': ['XJ', 'QH', 'NM', 'NX', 'SX'],
#              'GX': ['YN', 'GZ', 'HN', 'GD'],
#              'GZ': ['YN', 'SC', 'CQ', 'HN', 'GX'],
#              'HA': ['HB', 'SN', 'SX', 'HE', 'SD', 'AH'],
#              'HB': ['HN', 'CQ', 'SN', 'HA', 'AH', 'JX'],
#              'HE': ['BJ', 'TJ', 'LN', 'NM', 'SX', 'HA', 'SD'],
#              'HI': [],
#              'HK': ['GD'],
#              'HL': ['NM', 'JL'],
#              'HN': ['GX', 'GZ', 'CQ', 'HB', 'JX'],
#              'JL': ['HL', 'NM', 'LN'],
#              'JS': ['SD', 'AH', 'ZJ'],
#              'JX': ['GD', 'HN', 'HB', 'AH', 'ZJ', 'FJ'],
#              'LN': ['JL', 'NM', 'HE'],
#              'NM': ['GS', 'NX', 'SN', 'HB', 'LN', 'JL', 'HL'],
#              'NX': ['GS', 'NM', 'SN'],
#              'QH': ['XJ', 'XZ', 'GS', 'SC'],
#              'SC': ['XZ', 'QH', 'GS', 'SN', 'CQ', 'GZ', 'YN'],
#              'SD': ['HE', 'HA', 'AH', 'JS'],
#              'SH': ['JS', 'ZJ'],
#              'SN': ['GS', 'NM', 'SX', 'HN', 'HB', 'CQ', 'SC', 'NX'],
#              'SX': ['SN', 'NM', 'HE', 'HA', 'CQ', 'SC'],
#              'TJ': ['BJ', 'HE'],
#              'TW': [],
#              'XJ': ['GS', 'QH', 'XZ'],
#              'XZ': ['XJ', 'QH', 'SC', 'YN'],
#              'YN': ['XZ', 'SC', 'GZ', 'GX'],
#              'ZJ': ['SH', 'AH', 'JS', 'JX', 'FJ'],
#
# }

variables = provinces.keys()

domains = {}
for v in variables:
    domains[v] = ['R', 'G', 'B', 'Y']

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

colorChina = csp.CSP(variables, domains, provinces, constraints)
colorChina.label = 'China'
myCSPs = [
     {
        'csp' : colorChina,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : colorChina,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : colorChina,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : colorChina,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : colorChina,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : colorChina,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]