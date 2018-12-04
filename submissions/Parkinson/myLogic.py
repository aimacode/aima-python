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

# jedi = {
#     'kb': '''
# Jedi(Anakin)
# Father(Anakin, Luke)
# (Jedi(x)) ==> Midichlorians(x)
# (Father(x, y) & Jedi(x)) ==> Midichlorians(y)
#
# ''',
#     'queries':'''
# Jedi(x)
# Midichlorians(x)
# ''',
# }

zoo = {
    'kb': '''
Ostrich(Ozzy)
Deer(Dave)
Tiger(Tony)
Father(Tony, Thomas)
(Ostrich(x) ==> Bird(x))  
(Tiger(x) ==> Mammal(x))
(Deer(x) ==> Prey(x))
(Tiger(x) & Prey(y) ==> Eats(x, y))
(Father(x, y) & Tiger(x) ==> Mammal(y))

''',
    'queries':'''
Bird(x)
Mammal(x)  
Prey(x)
Eats(x, y)  
'''
}

Examples = {
    'zoo': zoo,
}
