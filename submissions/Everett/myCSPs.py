import csp

rgbp = ['R', 'G', 'B', 'P']

# The 23 wards of Tokyo

d2 = {'ADACHI' : rgbp, 'KATSU' : rgbp, 'EDOG' : rgbp, 'KOTO' : rgbp, 'CHUO': rgbp, 'SUMI': rgbp, 'TAITO':  rgbp, 'ARAK':   rgbp, 'KITA':   rgbp, 'BUNK':   rgbp, 'CHIY':   rgbp, 'MINA':   rgbp, 'SHIN':   rgbp,
    'SHINA':  rgbp,
    'OTA':    rgbp,
    'MEGU':   rgbp,
    'SETA':   rgbp,
    'SUGI':   rgbp,
    'NAKA':   rgbp,
    'NERI':   rgbp,
    'ITA':    rgbp,
    'TOSHI':  rgbp,
    'SHIB' :  rgbp,
      }

v2 = d2.keys()

n2 = {'ADACHI' : ['KATSU','SUMI','ARAK','KITA'],
      'KATSU' : ['ADACHI', 'EDOG','SUMI'],
      'EDOG' : ['KATSU', 'KOTO','SUMI'],
      'KOTO' : ['EDOG','CHUO','SUMI'],
      'CHUO':['KOTO','SUMI','MINA','CHIY','TAITO'],
      'TAITO':['CHUO','SUMI','CHIY','BUNK','KITA','ARAK'],
      'SUMI':['KATSU','EDOG','KOTO','CHUO','TAITO','ADACHI','ARAK'],
      'ARAK': ['TAITO', 'BUNK', 'SUMI', 'ADACHI', 'KITA'],
      'KITA': ['ADACHI', 'ARAK', 'BUNK', 'TOSHI', 'ITA','TAITO'],
      'BUNK': ['ARAK', 'TAITO', 'CHIY', 'SHIN', 'TOSHI', 'KITA'],
      'CHIY': ['CHUO', 'MINA', 'SHIN', 'BUNK', 'TAITO'],
      'MINA': ['CHUO', 'CHIY', 'SHIN', 'SHIB', 'SHINA'],
      'SHIN': ['MINA', 'CHIY', 'BUNK', 'TOSHI', 'NERI', 'NAKA', 'SHIB'],
      'SHINA': ['MINA', 'SHIB', 'MEGU', 'OTA'],
      'OTA': ['SHINA', 'MEGU', 'SETA'],
      'MEGU': ['SHIB', 'MINA', 'SHINA', 'OTA', 'SETA', ],
      'SETA': ['OTA', 'MEGU', 'SHIB', 'SUGI'],
      'SUGI': ['SETA', 'NERI', 'NAKA', 'SHIB'],
      'NAKA': ['SUGI', 'SHIB', 'SHIN', 'TOSHI', 'NERI'],
      'NERI': ['TOSHI', 'NAKA', 'SUGI', 'SHIN', 'ITA'],
      'ITA': ['KITA', 'NERI', 'TOSHI'],
      'TOSHI': ['KITA', 'BUNK', 'ITA', 'NERI', 'NAKA', 'SHIN'],
      'SHIB': ['NAKA', 'SHIN', 'SUGI', 'SETA', 'MEGU', 'SHINA', 'MINA'],}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

c2 = csp.CSP(v2, d2, n2, constraints)
c2.label = 'Map of Japan'

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
        #'order_domain_values': csp.lcv,
        'inference': csp.mac,
        #'inference': csp.forward_checking,
    },


]
