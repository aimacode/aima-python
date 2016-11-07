# farmer = {
#     'kb': '''
# Farmer(Mac)
# Rabbit(Pete)
# Mother(MrsMac, Mac)
# Mother(MrsRabbit, Pete)
# (Rabbit(r) & Farmer(f)) ==> Hates(f, r)
# (Mother(m, c)) ==> Loves(m, c)
# (Mother(m, r) & Rabbit(r)) ==> Rabbit(m)
# (Farmer(f)) ==> Human(f)
# (Mother(m, h) & Human(h)) ==> Human(m)
# ''',
pets = {
    'kb': '''
Boy(Dan)
Girl(Sally)
Dog(Spot)
Cat(Fluffy)
Mouse(Jerry)
Pet(Spot, Fluffy)
Owner(Dan, Sally, Spot, Fluffy)
Owner(Sally, Fluffy)
Treat(Cheese)
(Cat(c) & Dog(d)) ==> Hates(c, d)
(Cat(c) & Mouse(m)) ==> Loves(c, m)
(Boy(b) & Girl(g)) ==> Loves(b, g)
(Mouse(m) & Cat(c) & Dog(d)) ==> Wants(Cheese(c))
Mouse(m) ==> Loves(Cheese(c))
Boy(b) ==> Buys(Dog(d))
Girl(g) ==> Buys(Cat(c))
Boy(b) ==> Eats(Cheese(c))
Boy(b) & Girl(g) ==> Loves(b, g)
Girl(g) & Boy(b) ==> Hates(g, b)
Mouse(m) ==> Gets(Cheese(c))
''',




# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''

Hates(x, y)
Loves(x, y)
Wants(x)
Eats(x)
Buys(x)
Gets(x)
''',
#    'limit': 1,
}

# weapons = {
#     'kb': '''
# (American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)
# Owns(Nono, M1)
# Missile(M1)
# (Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)
# Missile(x) ==> Weapon(x)
# Enemy(x, America) ==> Hostile(x)
# American(West)
# Enemy(Nono, America)
# ''',
#     'queries':'''
# Criminal(x)
# ''',
# }

Examples = {
     'pets': pets,
#     'weapons': weapons,
}