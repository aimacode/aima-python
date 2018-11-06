import csp

rgb = ['blue ', 'red', 'purple', 'orange']

#colors = { 'A' : rgb, 'B' : rgb, 'C' : ['R'], 'D' : rgb,}

d2 = { 'JunkJunction' : rgb,
           'HauntedHills' : rgb,
           'SnobbyShores' : rgb,
           'GreasyGrove' : rgb,
           'PleasantPark' : rgb,
           'LazyLinks' : rgb,
           'LootLake' : rgb,
           'TiltedTowers' : rgb,
           'ShiftyShafts' : rgb,
           'FlushFactory' : rgb,
           'RiskyReels' : rgb,
           'DustyDivot' : rgb,
           'SaltySprings' : rgb,
           'FatalFields' : rgb,
           'WailingWoods' : rgb,
           'RetailRow' : rgb,
           'LonleyLodge' : rgb,
           'ParadisePalms' : rgb}

v2 = d2.keys()

# zone = {'A' : ['B', 'C', 'D'],
#       'B' : ['A', 'C', 'D'],
#       'C' : ['A', 'B'],
#       'D' : ['A', 'B'],}

n2 = {'JunkJunction' :   ['HauntedHills', 'PleasantPark'],
        'HauntedHills' :   ['JunkJunction', 'PleasantPark', 'SnobbyShores'],
        'SnobbyShores' :   ['HauntedHills', 'PleasantPark', 'GreasyGrove'],
        'GreasyGrove'  :   ['SnobbyShores', 'PleasantPark', 'TiltedTowers', 'ShiftyShafts', 'FlushFactory'],
        'PleasantPark' :   ['JunkJunction', 'HauntedHills', 'SnobbyShores', 'GreasyGrove', 'TiltedTowers', 'LootLake', 'LazyLinks'],
        'LazyLinks'    :   ['PleasantPark', 'LootLake', 'DustyDivot', 'RiskyReels'],
        'LootLake'     :   ['PleasantPark', 'TiltedTowers', 'DustyDivot', 'LazyLinks'],
        'TiltedTowers' :   ['PleasantPark', 'GreasyGrove', 'ShiftyShafts', 'SaltySprings', 'DustyDivot', 'LootLake'],
        'ShiftyShafts' :   ['TiltedTowers', 'GreasyGrove', 'FlushFactory', 'FatalFields', 'SaltySprings'],
        'FlushFactory' :   ['GreasyGrove', 'ShiftyShafts', 'FatalFields', 'ParadisePalms'],
        'RiskyReels'   :   ['LazyLinks', 'DustyDivot', 'WailingWoods'],
        'DustyDivot'   :   ['RiskyReels', 'LazyLinks', 'LootLake', 'TiltedTowers', 'SaltySprings', 'RetailRow', 'WailingWoods'],
        'SaltySprings' :   ['DustyDivot', 'TiltedTowers', 'ShiftyShafts', 'FatalFields', 'ParadisePalms', 'LonleyLodge', 'RetailRow'],
        'FatalFields'  :   ['SaltySprings', 'ShiftyShafts', 'FlushFactory', 'ParadisePalms'],
        'WailingWoods' :   ['RiskyReels', 'DustyDivot', 'RetailRow', 'LonleyLodge'],
        'RetailRow'    :   ['WailingWoods', 'DustyDivot', 'SaltySprings', 'LonleyLodge'],
        'LonleyLodge'  :   ['WailingWoods', 'RetailRow', 'SaltySprings', 'ParadisePalms'],
        'ParadisePalms':   ['LonleyLodge', 'SaltySprings', 'FatalFields', 'FlushFactory'],
      }


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

fortnite = csp.CSP(v2, d2, n2, constraints)
fortnite.label = 'Fortnite'

myCSPs = [
    {
        'csp' : fortnite,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : fortnite,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : fortnite,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : fortnite,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : fortnite,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : fortnite,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
