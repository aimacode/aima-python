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

animals = {
'Differ': '''
Ben,Jerry
''',
    'kb': '''
(TwoLegs(x) & Wings(x) & Mammal(x) & Eats(x,y) & Insect(y)) ==> Bat(x)
Beetle(Jerry)
Beetle(x) ==> Insect(x)
(Insect(x) & Eats(y,x)) ==> Prey(x)
Insect(x) ==> Animal(x)
Eats(Ben, Jerry)
Wings(x) ==> TwoLegs(x)
WarmBlooded(Ben)
Vertebrate(Ben)
(WarmBlooded(x) & Vertebrate(x)) ==> Mammal(x)
Wings(Ben)

''',
    'queries':'''
Insect(x)
TwoLegs(x)
Mammal(x)
Animal(x)
Prey(x)
Bat(x)
''',
    'limit': 5
}

Examples = {
    # 'farmer': farmer,
    # 'weapons': weapons,
    'animals': animals
}