import csp
rgb = ['R', 'G', 'B']

domains = {
    'Aosta Valley': rgb,
    'Piedmont': rgb,
    'Liguria': rgb,
    'Lombardy': rgb,
    'Trentino': rgb,
    'South Tyrol': rgb,
    'Veneto': rgb,
    'Friuli-Venezia Giulia': rgb,
    'Emilia-Romagna': rgb,
    'Tuscany': rgb,
    'Umbria': rgb,
    'Marche': rgb,
    'Lazio': rgb,
    'Abruzzo': rgb,
    'Molise': rgb,
    'Campania': rgb,
    'Apulia': rgb,
    'Basilicata': rgb,
    'Calabria': rgb,
}

neighbors = {
    'Aosta Valley': ['Piedmont'],
    'Piedmont': ['Liguria','Lombardy','Emilia-Romagna'],
    'Liguria': ['Piedmont','Emilia-Romagna','Tuscany'],
    'Lombardy': ['Piedmont','Emilia-Romagna','Veneto','Trentino','South Tyrol'],
    'Trentino': ['South Tyrol','Veneto','Lombardy'],
    'South Tyrol': ['Lombardy','Trentino','Veneto'],
    'Veneto': ['Friuli-Venezia Giulia','Trentino','South Tyrol','Lombardy','Emilia-Romagna'],
    'Friuli-Venezia Giulia': ['Veneto'],
    'Emilia-Romagna': ['Veneto','Lombardy','Tuscany','Liguria','Marche','Piedmont'],
    'Tuscany': ['Liguria','Emilia-Romagna','Marche','Umbria','Lazio'],
    'Umbria': ['Tuscany','Lazio','Marche'],
    'Marche': ['Emilia-Romagna','Tuscany','Umbria','Lazio','Abruzzo'],
    'Lazio': ['Tuscany','Umbria','Abruzzo','Molise','Campania'],
    'Abruzzo': ['Marche','Lazio','Molise'],
    'Molise': ['Abruzzo','Lazio','Campania','Apulia'],
    'Campania': ['Lazio','Molise','Apulia','Basilicata'],
    'Apulia': ['Molise','Campania','Basilicata'],
    'Basilicata': ['Apulia','Campania','Calabria'],
    'Calabria': ['Basilicata'],
}

vars = domains.keys()
domains = {}
for v in vars:
    domains[v] = ['R', 'G', 'B', 'P', 'O', 'T', 'M']

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myItalymap = csp.CSP(vars, domains, neighbors, constraints)

myCSPs = [
    {'csp': myItalymap,
     # 'select_unassigned_variable':csp.mrv,
     }
]