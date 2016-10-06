import csp

neighbors = {'AL': ['MS', 'TN', 'GA', 'FL'],
             'AR': ['OK', 'TX', 'MO', 'MS', 'TN', 'LA'],
             'AZ': ['CA', 'NV', 'UT'],
             'CA': ['OR', 'NV', 'AZ'],
             'CO': ['UT', 'WY', 'NE', 'KA', 'OK', 'NM'],
             'CT': ['NY', 'MA', 'RI'],
             'DC': ['VA', 'MD'],
             'DE': ['PA', 'NJ', 'MD'],
             'FL': ['AL', 'GA'],
             'GA': ['AL', 'TN', 'NC', 'SC', 'FL'],
             'IA': ['SD', 'NE', 'MN', 'WI', 'IL', 'MO'],
             'ID': ['WA', 'OR', 'NV', 'MT', 'WY', 'UT'],
             'IL': ['IA', 'MO', 'WI', 'IN', 'KY'],
             'IN': ['IL', 'OH', 'KY', 'MI'],
             'KA': ['CO', 'NE', 'MO', 'OK'],
             'KY': ['MO', 'IL', 'IN', 'OH', 'WV', 'VA', 'TN'],
             'LA': ['TX', 'AR', 'MS'],
             'MA': ['NY', 'VT', 'NH', 'RI', 'CT'],
             'MD': ['PA', 'WV', 'VA', 'DE', 'DC'],
             'ME': ['NH'],
             'MI': ['WI', 'OH', 'IN'],
             'MN': ['ND', 'SD', 'WI', 'IA'],
             'MO': ['NE', 'KA', 'OK', 'IA', 'IL', 'KY', 'TN', 'AR'],
             'MS': ['AR', 'LA', 'TN', 'AL'],
             'MT': ['ID', 'ND', 'SD', 'WY'],
             'NC': ['TN', 'GA', 'VA', 'SC'],
             'ND': ['MT', 'MN', 'SD'],
             'NE': ['WY', 'CO', 'SD', 'IA', 'MO', 'KA'],
             'NH': ['VT', 'MA', 'ME'],
             'NJ': ['PA', 'NY', 'DE'],
             'NM': ['CO', 'OK', 'TX'],
             'NV': ['OR', 'CA', 'ID', 'UT', 'AZ'],
             'NY': ['PA', 'VT', 'MA', 'CT', 'NJ'],
             'OH': ['IN', 'MI', 'PA', 'WV', 'KY'],
             'OK': ['CO', 'NM', 'KA', 'MO', 'AR', 'TX'],
             'OR': ['WA', 'ID', 'NV', 'CA'],
             'PA': ['OH', 'NY', 'NJ', 'DE', 'MD', 'WV'],
             'RI': ['MA', 'CT'],
             'SC': ['GA', 'NC'],
             'SD': ['MT', 'WY', 'ND', 'MN', 'IA', 'NE'],
             'TN': ['MO', 'AR', 'MS', 'AL', 'KY', 'VA', 'NC', 'GA'],
             'TX': ['NM', 'OK', 'AR', 'LA'],
             'UT': ['NV', 'ID', 'WY', 'CO', 'AZ'],
             'VA': ['KY', 'TN', 'WV', 'MD', 'DC', 'NC'],
             'VT': ['NY', 'NH', 'MA'],
             'WA': ['OR', 'ID'],
             'WI': ['MN', 'IA', 'MI', 'IL'],
             'WV': ['OH', 'KY', 'PA', 'MD', 'VA'],
             'WY': ['ID', 'UT', 'MT', 'SD', 'NE', 'CO'],
}

variables = neighbors.keys()

domains = {}
for v in variables:
    domains[v] = ['R', 'G', 'B', 'K']

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