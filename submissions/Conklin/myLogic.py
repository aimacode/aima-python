videoGameLogic = {
    'kb': '''
Protagonist(Geralt)
Protagonist(Tidus)
Protagonist(Dragonborn)
Protagonist(Dante)
Antagonist(Eredin)
Antagonist(Jecht)
Antagonist(Alduin)
Antagonist(Vergil)
Antagonist(Imlerith)
Antagonist(Caranthir)
Antagonist(Nithral)
Antagonist(Sin)
designatedAntagonist(Geralt, Eredin)
designatedAntagonist(Geralt, Imlerith)
designatedAntagonist(Geralt, Nithral)
designatedAntagonist(Tidus, Sin)
designatedAntagonist(Tidus, Jecht)
designatedAntagonist(Dragonborn, Alduin)
designatedAntagonist(Dante, Vergil)


Protagonist(p) & Protagonist(y) ==> goodGuys(p,y)
Protagonist(p) & Antagonist(a) ==> Enemies(p,a)
''',
    'queries':'''
    Protagonist(p)
    Antagonist(a)
    designatedAntagonist(p,a)
    Enemies(p,a)
    goodGuys(p,y)
''',
}

Examples = {
    'videoGameLogic': videoGameLogic,
}
