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

ocean = {
    'kb': '''
Fish(Gill)
Fish(Rocky)
Fish(Uma)
Dolphin(Delphine)
Dolphin(Dale)
Shrimp(Sam)
Shrimp(Dave)
Crab(Craig)
Crab(Chris)
Crab(x) ==> Crustacean(x)
Shrimp(x)  ==> Crustacean(x)
Predator(Gill, Sam)
Predator(Delphine, Gill)
Predator(Delphine, Shrimp)
Prey(Gill, Delphine)
Prey(Sam, Gill)
(Shrimp(s) & Fish(f)) ==> Eats(f, s)
(Fish(f) & Dolphin(d)) ==> Eats(d, f)
(Shrimp(f) & Dolphin(d)) ==> Eats(d, f)
Fish(f) & Shrimp(s) ==> Fears(s,f)






''',
    'queries':'''
Prey(x,y)
Eats(x, y)
Fish(x)
Fears(x,y)
Crustacean(x)
''',
}

Examples = {
    # 'farmer': farmer,
    # 'weapons': weapons,
    'ocean': ocean,
}