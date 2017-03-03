import csp
rgb = ['R', 'G', 'B', 'K']

domains = {
    #'Hokkaido': rgb,
    'Aomori': rgb,
    'Akita': rgb,
    'Iwate': rgb,
    'Yamagata': rgb,
    'Miyagi': rgb,
    'Fukushima': rgb,
    'Nigata': rgb,
    'Gunma': rgb,
    'Tochigi': rgb,
    'Ibaraki': rgb,
    'Chiba': rgb,
    'Saitama': rgb,
    #'Nagano': rgb,
    'Tokyo': rgb,
    'Kanagawa': rgb,
    #'Shizuoka': rgb,
    #'Yamanashi': rgb,
    #'Toyama': rgb,
    #'Gifu': rgb,
    #'Aichi': rgb,
    #'Ishikawa': rgb,
    #'Fukui': rgb,
    #'Shiga': rgb,
    #'Kyoto': rgb,
    #'Mie': rgb,
    #'Nara': rgb,
    #'Wakayama': rgb,
    #'Osaka': rgb,
    #'Hyogo': rgb,
    #'Tottori': rgb,
    #'Okayama': rgb,
    #'Hiroshima': rgb,
    #'Shimane': rgb,
    #'Yamaguchi': rgb,
    #'Kagawa': rgb,
    #'Tokushima': rgb,
    #'Kochi': rgb,
    #'Ehime': rgb,
    #'Fukuoka': rgb,
    #'Saga': rgb,
    #'Oita': rgb,
    #'Nagasaki': rgb,
    #'Kumamoto': rgb,
    #'Miyazaki': rgb,
    #'Kagoshima': rgb
}

variables = domains.keys()

neighbors = {
    #'Hokkaido': [],
    'Aomori': ['Akita', 'Iwate'],
    'Akita': ['Aomori', 'Iwate', 'Yamagata', 'Miyagi'],
    'Iwate': ['Aomori', 'Akita', 'Miyagi'],
    'Yamagata': ['Akita', 'Miyagi', 'Fukushima', 'Nigata'],
    'Miyagi': ['Iwate', 'Yamagata', 'Fukushima', 'Akita'],
    'Fukushima': ['Miyagi', 'Yamagata', 'Nigata', 'Tochigi', 'Ibaraki'],
    'Nigata': ['Yamagata', 'Fukushima', 'Gunma', 'Nagano', 'Toyama'],
    'Tochigi': ['Fukushima', 'Gunma', 'Ibaraki'],
    'Ibaraki': ['Tochigi', 'Saitama', 'Fukushima', 'Chiba'],
    'Chiba': ['Ibaraki', 'Saitama', 'Tokyo'],
    'Saitama': ['Gunma', 'Ibaraki', 'Chiba', 'Tokyo'],
    'Gunma': ['Saitama', 'Tochigi'],
    'Tokyo': ['Saitama', 'Chiba', 'Kanagawa'],
    'Kanagawa': ['Tokyo'],
    #'Nagano': ['Gunma', 'Nigata', 'Toyama', 'Gifu', 'Aichi', 'Shizuoka', 'Yamanashi'],
    #'Yamanashi': ['Kanagawa', 'Tokyo', 'Saitama', 'Nagano', 'Shizuoka'],
    #'Shizuoka': ['Yamanashi', 'Kanagawa', 'Nagano', 'Aichi'],
    #'Aichi':  ['Shizuoka', 'Nagano', 'Gifu', 'Mie'],
    #'Toyama': ['Nigata', 'Nagano', 'Gifu', 'Ishikawa'],
    #'Ishikawa': ['Toyama', 'Gifu', 'Fukui'],
    #'Gifu':['Toyama', 'Ishikawa', 'Fukui', 'Shiga', 'Mie', 'Aichi', 'Nagano'],
    #'Fukui':['Ishikawa', 'Gifu', 'Shiga', 'Kyoto'],
    #'Shiga': ['Fukui', 'Gifu', 'Mie', 'Kyoto'],
    #'Mie': ['Shiga', 'Nara', 'Wakayama', 'Aichi', 'Gifu'],
    #'Nara': ['Mie', 'Kyoto', 'Osaka', 'Wakayama'],
    #'Wakayama': ['Osaka', 'Nara', 'Mie'],
    #'Osaka': ['Hyogo', 'Kyoto', 'Nara', 'Wakayama'],
    #'Kyoto': ['Hyogo', 'Osaka', 'Nara', 'Shiga', 'Fukui'],
    #'Hyogo': ['Kyoto', 'Osaka', 'Okayama', 'Tottori'],
    #'Tottori': ['Hyogo', 'Okayama', 'Shimane'],
    #'Okayama': ['Hyogo', 'Tottori', 'Hiroshima'],
    #'Shimane': ['Tottori', 'Hiroshima', 'Yamaguchi'],
    #'Yamaguchi': ['Shimane', 'Hiroshima'],
    #'Hiroshima': ['Okayama', 'Shimane', 'Yamaguchi'],
    #'Kagawa': ['Tokushima', 'Ehime'],
    #'Tokushima': ['Kagawa', 'Ehime', 'Kochi'],
    #'Ehime': ['Kagawa', 'Tokushima', 'Kochi'],
    #'Kochi': ['Ehime', 'Tokushima'],
    #'Fukuoka': ['Oita', 'Kumamoto', 'Saga'],
    #'Oita': ['Fukuoka', 'Kumamoto', 'Miyazaki'],
    #'Miyazaki': ['Oita', 'Kumamoto', 'Kagoshima'],
    #'Kagoshima': ['Miyazaki', 'Kumamoto'],
    #'Kumamoto': ['Kagoshima', 'Miyazaki', 'Oita', 'Fukuoka'],
    #'Saga': ['Nagasaki', 'Fukuoka'],
    #'Nagasaki': ['Saga']
}

def constraints(A, a, B, b):
    if A == B:      # e.g. NSW == NSW
        return True

    if a == b:      # e.g. WA = G and SA = G
        return False

    return True

myJap = csp.CSP(variables, domains, neighbors, constraints)

myCSPs = [
    {'csp': myJap,
     # 'select_unassigned_variable':csp.mrv,
     }
]
