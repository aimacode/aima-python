import csp

rgb = {'green', 'orange', 'pink', 'yellow', 'red', 'purple', 'brown', 'cream', 'blue', 'BLELLOW'}

colors = {'BAVARIA': rgb, 'WURTTEMBERG': rgb, 'THURINGEN': rgb, 'HESSEN': rgb,
          'RHEINLAND': rgb, 'SAXONY': rgb, 'ANHALT': rgb, 'BRANDENBURG': rgb,
          'WESTFALEN': rgb, 'LOWER-SAXONY': rgb, 'HOLSTEIN': rgb, 'VORPOMMERN': rgb,
          'SAARLAND': rgb}

cities = colors.keys()

map = {'BAVARIA': ['WURTTEMBERG', 'HESSEN', 'THURINGEN', 'SAXONY'],
       'WURTTEMBERG': ['BAVARIA', 'HESSEN', 'RHEINLAND'],
       'THURINGEN': ['BAVARIA', 'SAXONY', 'ANHALT', 'HESSEN', 'LOWER-SAXONY'],
       'HESSEN': ['THURINGEN', 'WURTTEMBERG', 'RHEINLAND', 'WESTFALEN', 'LOWER-SAXONY'],
       'RHEINLAND': ['SAARLAND', 'WURTTEMBERG', 'WESTFALEN'],
       'SAXONY': ['THURINGEN', 'ANHALT', 'BRANDENBURG', 'BAVARIA'],
       'ANHALT': ['SAXONY', 'BRANDENBURG', 'THURINGEN', 'LOWER-SAXONY'],
       'BRANDENBURG': ['VORPOMMERN', 'ANHALT', 'SAXONY', 'LOWER-SAXONY'],
       'WESTFALEN': ['LOWER-SAXONY', 'HESSEN', 'RHEINLAND'],
       'LOWER-SAXONY': ['HOLSTEIN', 'ANHALT', 'WESTFALEN', 'HESSEN', 'THURINGEN', 'BRANDENBURG', 'VORPOMMERN'],
       'HOLSTEIN': ['VORPOMMERN', 'LOWER-SAXONY'],
       'VORPOMMERN': ['BRANDENBURG', 'LOWER-SAXONY', 'HOLSTEIN'],
       'SAARLAND': ['RHEINLAND']
       }


def constraints(A, a, B, b):
    if A == B:  # e.g. NSW == NSW
        return True

    if a == b:  # e.g. WA = G and SA = G
        return False

    return True


germany = csp.CSP(cities, colors, map, constraints)
germany.label = 'GERMANY'

myCSPs = [
    {
        'csp': germany,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    }
]
