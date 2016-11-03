farmer = {
    'kb': '''
Farmer(Mac)
Rabbit(Pete)
Mother(MrsMac, Mac)
Mother(MrsRabbit, Pete)
(Rabbit(r) & Farmer(f)) ==> Hates(f, r)
(Mother(m, c)) ==> Loves(m, c)
(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)
(Farmer(f)) ==> Human(f)
(Mother(m, h) & Human(h)) ==> Human(m)
''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Human(x)
Hates(x, y)
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

cocktails = {'kb': '''
    Liquor(Vodka)
    Liquor(Tequila)
    Liquor(Gin)
    Liquor(Whiskey)
    Liquor(Rum)
    Mixer(OrangeJuice)
    Mixer(Coke)
    Mixer(MargaritaMix)
    Mixer(BloodMaryMix)
    Mixer(GingerAle)
    Cocktail(Vodka, OrangeJuice)
    Cocktail(Vodka, BloodyMaryMix)
    Cocktail(Vodka, GingerAle)
    Cocktail(Tequila, OrangeJuice)
    Cocktail(Tequila, MargaritaMix)
    Cocktail(Tequila, BloodMaryMix)
    Cocktail(Whiskey, Coke)
    Cocktail(Whiskey, GingerAle)
    Cocktail(Rum, OrangeJuice)
    Cocktail(Rum, Coke)

    (Liquor(l) & Mixer(m)) ==> Cocktail(l, m)
    (Mixer(m)) ==> Nonalcoholic(m)
    (Liquor(l)) ==> Alcoholic(l)
    (Mixer(m)) ==> Drink(m)
    (Liquor(l)) ==> Drink(m)

''',

    'queries': '''
        Liquor(x)
        Cocktail(x, y)
    ''',
}



Examples = {
    'cocktails': cocktails,
}