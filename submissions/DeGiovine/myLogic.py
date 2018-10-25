vehicles = {
    'kb': '''
Company(Lexus)
Company(Chevy)
Company(Ducati)
Company(Lamborghini)
Company(Volkswagen)

Parent(Toyota, Lexus)
Parent(GM, Chevy)
Parent(Lamborghini, Ducati)
Parent(Volkswagen, Lamborghini)

Product(Car, Lexus)
Product(Car, Chevy)
Product(Bike, Ducati)
Product(Car, Lamborghini)
Product(Car, Volkswagen)

(Company(c) & Product(p)) ==> Sells(p, c)
(Parent(p,s) ==> ParentCompany(p,s)
(Car(a)) ==> Vehicle(v)   
 
    ''',
    'queries':'''
Company(y)
ParentCompany(x, y)
''',
}

# weights2d = np.array([
#    [1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1],


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

Examples = {
    'farmer': farmer,
    'weapons': weapons,
    'vehicles': vehicles,
}
