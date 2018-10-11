import csp

rgby = ['R', 'G', 'B', 'Y']

d2 = {850: rgby, 386 : rgby, 904 : rgby, 352 : rgby, 407 : rgby, 321 : rgby, 727 : rgby, 813 : rgby, 863 : rgby, 772 : rgby, 941 : rgby, 561 : rgby, 239 : rgby, 754 : rgby, 305 : rgby, }

v2 = d2.keys()

n2 = {850 : [386, 352],
      386 : [850, 904, 352, 407, 321],
      904 : [352, 386],
      352 : [850, 386, 904, 727, 813, 863, 407],
      407 : [386, 352, 321, 772, 863],
      321 : [386, 407, 772],
      727 : [352, 813],
      813 : [727, 352, 863, 941],
      941 : [813, 863, 239],
      239 : [941, 863, 754, 305],
      305 : [239, 754],
      754 : [239, 305, 561, 863],
      561 : [754, 863, 772],
      772 : [561, 863, 407, 321],
      863 : [352, 407, 772, 561, 754, 239, 941, 813]
      }

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

FLAreaCodes = csp.CSP(v2, d2, n2, constraints)
FLAreaCodes.label = 'Florida Area Codes'

myCSPs = [
    {
        'csp' : FLAreaCodes,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : FLAreaCodes,
        'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : FLAreaCodes,
        # 'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : FLAreaCodes,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
    {
        'csp' : FLAreaCodes,
        # 'select_unassigned_variable': csp.mrv,
        # 'order_domain_values': csp.lcv,
        # 'inference': csp.mac,
        'inference': csp.forward_checking,
    },
    {
        'csp' : FLAreaCodes,
        'select_unassigned_variable': csp.mrv,
        'order_domain_values': csp.lcv,
        'inference': csp.mac,
        # 'inference': csp.forward_checking,
    },
]
