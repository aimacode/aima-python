import csp

rgb = ['R', 'G', 'B','P','T']

# The 22 wards of Tokyo
domains = {
    'ADACHI': rgb,
    'KATSU':  rgb,
    'EDOG':   rgb,
    'KOTO':   rgb,
    'CHUO':   rgb,
    'SUMI':   rgb,
    'TAITO':  rgb,
    'ARAK':   rgb,
    'KITA':   rgb,
    'BUNK':   rgb,
    'CHIY':   rgb,
    'MINA':   rgb,
    'SHIN':   rgb,
    'SHINA':  rgb,
    'OTA':    rgb,
    'MEGU':   rgb,
    'SETA':   rgb,
    'SUGI':   rgb,
    'NAKA':   rgb,
    'NERI':   rgb,
    'ITA':    rgb,
    'TOSHI':  rgb,
    'SHIB' :  rgb,

}

variables = domains.keys()

neighbors = {
    'ADACHI': ['KATSU','SUMI','ARAK','KITA'],
    'KATSU': ['ADACHI','SUMI','EDOG'],
    'EDOG':  ['KATSU','SUMI','KOTO'],
    'KOTO': ['SUMI','EDOG','CHUO'],
    'CHUO': ['MINA','KOTO','CHIY','SUMI','TAITO'],
    'SUMI': ['KATSU','EDOG','KOTO','CHUO','TAITO','ARAK','ADACHI'],
    'TAITO': ['CHUO','SUMI','CHIY','BUNK','KITA','ARAK'],
    'ARAK': ['TAITO','BUNK','SUMI','ADACHI','KITA'],
    'KITA':  ['ADACHI','ARAK','BUNK','TOSHI','ITA'],
    'BUNK': ['ARAK','TAITO','CHIY','SHIN','TOSHI','KITA'],
    'CHIY': ['CHUO','MINA','SHIN','BUNK','TAITO'],
    'MINA': ['CHUO','CHIY','SHIN','SHIB','SHINA'],
    'SHIN': ['MINA','CHIY','BUNK','TOSHI','NERI','NAKA','SHIB'],
    'SHINA': ['MINA','SHIB','MEGU','OTA'],
    'OTA': ['SHINA','MEGU','SETA'],
    'MEGU': ['SHIB','MINA','SHINA','OTA','SETA',],
    'SETA': ['OTA','MEGU','SHIB','SUGI'],
    'SUGI': ['SETA','NERI','NAKA','SHIB'],
    'NAKA': ['SUGI','SHIB','SHIN','TOSHI','NERI'],
    'NERI': ['TOSHI','NAKA','SUGI','SHIN','ITA'],
    'ITA': ['KITA','NERI','TOSHI'],
    'TOSHI': ['KITA','BUNK','ITA','NERI','NAKA','SHIN'],
    'SHIB': ['NAKA','SHIN','SUGI','SETA','MEGU','SHINA','MINA'],

}


def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True


myJapan = csp.CSP(variables, domains, neighbors, constraints)
#bob = csp.mrv(5, myJapan)
#inference = csp.mac()
#okiedokie=csp.forward_checking()

myCSPs = [
    {'csp': myJapan,
     # 'select_unassigned_variable':csp.mrv,
    # 'heurisic': bob,
     }
]