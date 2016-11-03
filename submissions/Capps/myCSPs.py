import csp

rgb = ['R', 'G', 'B']

domains = {
    'AL': rgb,
    'AZ': rgb,
    'AR': rgb,
    'CA': rgb,
    'CO': rgb,
    'CT': rgb,
    'DE': rgb,
    'FL': rgb,
    'GA': rgb,
    'ID': rgb,
    'IL': rgb,
    'IN': rgb,
    'IA': rgb,
    'KS': rgb,
    'KY': rgb,
    'LA': rgb,
    'ME': rgb,
    'MD': rgb,
    'MA': rgb,
    'MI': rgb,
    'MN': rgb,
    'MS': rgb,
    'MO': rgb,
    'MT': rgb,
    'NE': rgb,
    'NV': rgb,
    'NH': rgb,
    'NJ': rgb,
    'NM': rgb,
    'NY': rgb,
    'NC': rgb,
    'ND': rgb,
    'OH': rgb,
    'OK': rgb,
    'OR': rgb,
    'PA': rgb,
    'RI': rgb,
    'SC': rgb,
    'SD': rgb,
    'TN': rgb,
    'TX': rgb,
    'UT': rgb,
    'VT': rgb,
    'VA': rgb,
    'WA': rgb,
    'WV': rgb,
    'WI': rgb,
    'WY': rgb,
}

variables = domains.keys()

neighbors = {
    'AL': ['MS', 'GA', 'FL'],
    'AZ': ['CA', 'NV', 'UT', 'NM'],
    'AR': ['OK', 'TX', 'LA', 'MS', 'TN', 'MO'],
    'CA': ['OR', 'NV', 'AZ'],
    'CO': ['UT', 'NM', 'OK', 'KS', 'NE', 'WY'],
    'CT': ['RI', 'MA', 'NY'],
    'DE': ['PA', 'MD', 'NJ'],
    'FL': ['GA', 'AL'],
    'GA': ['SC', 'NC', 'TN', 'AL', 'FL'],
    'ID': ['WA', 'OR', 'NV', 'UT', 'WY', 'MT'],
    'IL': ['IA', 'WI', 'MO', 'IN', 'KY'],
    'IN': ['MI', 'OH', 'KY', 'IL'],
    'IA': ['WI', 'MN', 'SD', 'NE', 'MO', 'IL'],
    'KS': ['NE', 'CO', 'OK', 'MS'],
    'KY': ['TN', 'VA', 'WV', 'OH', 'IN', 'IL', 'MO'],
    'LA': ['MS', 'AR', 'TX'],
    'ME': ['NH'],
    'MD': ['VA', 'WV', 'PA'],
    'MA': ['NH', 'VT', 'RI', 'CT', 'NY'],
    'MI': ['OH', 'IN', 'WI'],
    'MN': ['WI', 'IA', 'ND', 'SD'],
    'MS': ['LA', 'AR', 'TN', 'AL'],
    'MO': ['KY', 'IL', 'IA', 'NE', 'KS', 'OK'],
    'MT': ['ID', 'WY', 'SD', 'ND'],
    'NE': ['SD', 'WY', 'CO', 'KS', 'MO', 'IA'],
    'NV': ['OR', 'CA', 'AZ', 'UT', 'ID'],
    'NH': ['MA', 'ME', 'VT'],
    'NJ': ['NY', 'PA', 'DE'],
    'NM': ['AZ', 'CO', 'TX', 'OK'],
    'NY': ['CT', 'VT', 'MA', 'NJ', 'PA'],
    'NC': ['VA', 'TN', 'SC', 'GA'],
    'ND': ['MT', 'SD', 'MN'],
    'OH': ['MI', 'IN', 'KY', 'WV', 'PA'],
    'OK': ['KS', 'CO', 'NM', 'TX', 'AR', 'MO'],
    'OR': ['WA', 'ID'],
    'PA': ['NY', 'OH', 'WV', 'MD', 'NJ', 'DE'],
    'RI': ['MA', 'CT'],
    'SC': ['NC', 'GA'],
    'SD': ['MT', 'WY', 'NE', 'IA', 'MN', 'ND'],
    'TN': ['MO', 'AR', 'MS', 'AL', 'GA', 'NC', 'VA', 'KY'],
    'TX': ['NM', 'OK', 'LA'],
    'UT': ['ID', 'NV', 'AZ', 'CO', 'WY'],
    'VT': ['NY', 'NH', 'MA'],
    'VA': ['WV', 'MD', 'KY', 'TN', 'NC'],
    'WA': ['ID', 'OR'],
    'WV': ['PA', 'OH', 'KY', 'VA', 'MD'],
    'WI': ['MI', 'MI', 'IA', 'IL'],
    'WY': ['MT', 'ID', 'UT', 'CO', 'NE', 'SD'],

}

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