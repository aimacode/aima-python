import csp

rgb = ['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Pink', 'Brown','Purple', 'White', 'Black', 'Gray','Teal','Maroon' ]


d2 = {
       'Jouf': rgb,
       'Tabuk': rgb, 'NorthernBorder': rgb,
       'Hail': rgb, 'Madinah': rgb, 'Qasim': rgb, 'Makkah': rgb,
       'Riyadh': rgb, 'EasternProvince': rgb, 'Baha': rgb, 'Asir': rgb,
       'Jizan': rgb, 'Najran': rgb, 'Yemen': rgb, 'Oman': rgb, 'UAE': rgb


       }

v2 = d2.keys()



n2 = {
       'Jouf' : ['Tabuk', 'Hail', 'NorthernBorder'],
       'Tabuk' : [ 'Madinah', 'Hail', 'Jouf'],
       'NorthernBorder' : ['Jouf', 'Hail','Riyadh', 'EasternProvince'],
       'Hail' : ['Jouf', 'Tabuk', 'Madinah', 'Qasim', ],
       'Madinah': ['Tabuk', 'Hail', 'Qasim', 'Makkah','Riyadh' ],
       'Qasim': ['Hail','Madinah','Riyadh'],
       'Makkah': ['Madinah', 'Riyadh', 'Baha', 'Asir', 'Jizan'],
       'Riyadh': ['Qasim', 'Madinah', 'Makkah', 'NorthernBorder', 'EasternProvince', 'Asir', 'Najran'],
       'EasternProvince': ['Riyadh','NorthernBorder', 'Najran', 'Yemen', 'Oman', 'UAE'],
       'Baha': [ 'Makkah','Asir'],
       'Asir': ['Baha', 'Riyadh', 'Makkah', 'Jizan', 'Najran', 'Yemen'],
       'Jizan': ['Asir','Yemen', 'Makkah'],
       'Najran': ['EasternProvince', 'Riyadh', 'Asir','Yemen'],
       'Yemen': ['Jizan','Asir','Najran', 'EasternProvince', 'Oman'],
       'Oman': ['EasternProvince', 'Yemen', 'UAE'],
       'UAE': ['EasternProvince', 'Oman'],




      }

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Really Lame'

myCSPs = [
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : c2,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
