farmer = {
    'kb': '''

Ocean(Atlantic)
Ocean(Pacific)
Tributary(Atlantic, Albemarle)
Tributary(Albemarle, Chowan)
Tributary(Chowan, Meherrin)
Tributary(Meherrin, WorrellMill)
Tributary(Pacific, Columbia)
Tributary(Columbia, Cowlitz)
Tributary(Cowlitz, Toutle)
Tributary(Toutle, Green)
Tributary(Atlantic, Gulf)
Tributary(Gulf, Mississippi)
Tributary(Mississippi, Red)
Tributary(Mississippi, Ohio)
Tributary(Ohio, Tennessee)
Tributary(Tennessee, Cumberland)
Tributary(Cumberland, Harpeth)
Tributary(Harpeth, BigTurnbull)
Tributary(BigTurnbull, SullivanBranch)
Tributary(BigTurnbull, TelleyBranch)
Tributary(BigTurnbull, LakeWeona)
Tributary(BigTurnbull, Turnbull)
Tributary(BigTurnbull, SullivanBranch)
Tributary(Cumberland, Mansker)
Tributary(Mansker, CenterPoint)
Tributary(Cumberland, Mansker)
Tributary(Mansker, Madison)
Tributary(Madison, Willis)
Tributary(Madison, Pattons)
Tributary(Mississippi, BigMuddy)
Tributary(BigMuddy, Cedar)
Tributary(Cedar, Bear)
Tributary(Mississippi, Ohio)

Tributary(w, x) & Tributary(x, y) ==> SeparatedTributary(w, y)
SeparatedTributary(w, x) & SeparatedTributary(w, y) ==> NotTouching(x, y)
Tributary(w, x) & Ocean(w) ==> Estuary(x)
Tributary(g, h) & Estuary(g) ==> SecondarySource(h)

''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Tributary(u, i)
Estuary(f)
SecondarySource(z)
''',
#    'limit': 1,
}

weapons = {
    'kb': '''
(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)
Owns(Nono, M1)
Missile(M1)
(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)
Missile(x) ==> Weapon(x)
Enemy(x, America) ==> Hostile(x)
American(West)
Enemy(Nono, America)
''',
    'queries':'''
Criminal(x)
''',
}

Examples = {
    'farmer': farmer,
    'weapons': weapons,
}