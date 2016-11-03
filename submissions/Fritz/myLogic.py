farmer = {
    'kb': '''

Ocean(Atlantic)
Ocean(Pacific)
Tributary(Atlantic, Brenin)
Tributary(Brenin, Cober)
Source(Cober)
Tributary(Atlantic, Albemarle)
Tributary(Albemarle, Chowan)
Tributary(Chowan, Meherrin)
Tributary(Meherrin, WorrellMill)
Source(WorrelMill)
Tributary(Pacific, Columbia)
Tributary(Columbia, Cowlitz)
Tributary(Cowlitz, Toutle)
Tributary(Toutle, Green)
Source(Green)
Tributary(Atlantic, Gulf)
Tributary(Gulf, Mississippi)
Tributary(Mississippi, Red)
Tributary(Mississippi, Ohio)
Tributary(Ohio, Tennessee)
Tributary(Tennessee, Cumberland)
Tributary(Cumberland, Harpeth)
Tributary(Harpeth, BigTurnbull)
Tributary(BigTurnbull, SullivanBranch)
Source(SullivanBranch)
Tributary(BigTurnbull, TelleyBranch)
Source(TelleyBranch)
Tributary(BigTurnbull, LakeWeona)
Source(LakeWeona)
Tributary(BigTurnbull, Turnbull)
Source(Turnbull)
Tributary(BigTurnbull, SullivanBranch)
Source(SullivanBranch)
Tributary(Cumberland, Mansker)
Tributary(Mansker, CenterPoint)
Source(CenterPoint)
Tributary(Cumberland, Mansker)
Tributary(Mansker, Madison)
Tributary(Madison, Willis)
Source(Willis)
Tributary(Madison, Pattons)
Source(Pattons)
Tributary(Mississippi, BigMuddy)
Tributary(BigMuddy, Cedar)
Tributary(Cedar, Bear)
Source(Bear)
Tributary(Pacific, Carnl)
Tributary(Carnl, Hara)
Source(Hara)


Tributary(w, x) & Tributary(x, y) ==> SeparatedTributary(w, y)
SeparatedTributary(w, x) & SeparatedTributary(w, y) ==> NotTouching(x, y)
Tributary(w, x) & Ocean(w) ==> Estuary(x)
Tributary(g, h) & Estuary(g) ==> SecondarySource(h)
Source(r) & SecondarySource(r) ==> ShortRiver(r)

''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Tributary(u, i)
Estuary(f)
Source(d)
ShortRiver(p)
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